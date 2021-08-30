import numpy as np
import random
from graph_model.building_graph import building_graph
from graph_model.utils import get_rest_working_time
from tqdm import tqdm


def load_dag(filename, batch_start):
    """
    读取DAG图
    :return:
    """
    G, time_cost, dig_to_table_name, list_batch_start = building_graph(filename, batch_start)
    return G, time_cost, dig_to_table_name, list_batch_start


def get_in_deg(G):
    """

    :param G: 边集, 用出度表示
    :return:
    """
    num_vertex = len(G)
    in_degs = [0] * num_vertex
    for out_edges in G:
        for out_vertex in out_edges:
            in_degs[out_vertex] += 1
    return in_degs


def topo_sort(G):
    """
    返回DAG图的一个拓扑排序结果
    :return:
    """
    num_vertex = len(G)
    in_vertex = [0] * num_vertex
    for out_edges in G:
        for out_vertices in out_edges:
            in_vertex[out_vertices] += 1
    queue = []
    for i in range(num_vertex):
        if in_vertex[i] == 0:
            queue.append(i)
    topo = []
    while queue:
        v = queue.pop()
        for out_vertex in G[v]:
            in_vertex[out_vertex] -= 1
            if in_vertex[out_vertex] == 0:
                queue.append(out_vertex)
        topo.append(v)
    if len(topo) < num_vertex:
        raise TypeError("Not a DAG")
    return topo


def transpose_G(G):
    """
    求图G的转置
    :param G:
    :return:
    """
    vertex_num = len(G)
    G_transpose = [[] for _ in range(vertex_num)]
    for i, edges in enumerate(G):
        for out_vertices in edges:
            G_transpose[out_vertices].append(i)
    return G_transpose


def distribute_task(hormone_i, mask, task_last_finish_time_on_server, last_finish_time_for_server, alpha, beta):
    """

    :param hormone_i:
    :param mask:
    :param task_last_finish_time_on_server: 任务分配到机器的最后完成时间
    :param last_finish_time_for_server: 在分配该任务前每个机器的所需执行时间
    :param alpha:
    :return:
    """
    candidate_tau = hormone_i.copy()
    candidate_tau[mask] = 0.0
    enta = np.max(task_last_finish_time_on_server) - task_last_finish_time_on_server + 1
    prob = (enta ** beta) * (candidate_tau ** alpha)

    if np.sum(prob) == 0.0:
        return int(np.where(np.random.multinomial(1, (1 - mask) / np.sum(1 - mask)) == 1)[0])

    prob = prob / np.sum(prob)  # 归一化
    server_chosen = int(np.where(np.random.multinomial(1, prob) == 1)[0])
    return np.where(last_finish_time_for_server == last_finish_time_for_server[server_chosen])[0].min()


def init_ant(queue, m, T, time_cost):
    """
    随机从队列中选m个任务,按照任务标号从小到大排序分配到服务器上
    :param queue:
    :param m:
    :return:
    """
    assert len(queue) >= m
    task_chosen = []
    for _ in range(m):
        task_chosen.append(queue.pop(random.randint(0, len(queue) - 1)))
    task_chosen.sort()
    last_finish_time_for_task = np.zeros(T)  # 任务的最晚完成时间
    last_finish_time_for_server = np.zeros(m)  # 当前服务器上的最晚完成时间
    server_distributed_for_task = -1 * np.ones(T).astype(np.uint8)  # 任务被分配于哪一台服务器
    working_order = [[] for _ in range(m)]
    for i, task in enumerate(task_chosen):
        last_finish_time_for_server[i] = time_cost[task]
        last_finish_time_for_task[task] = time_cost[task]
        server_distributed_for_task[task] = i
        working_order[i].append(task)
    distributed_time_for_each_server = last_finish_time_for_server.copy()
    total_distributed = np.sum(distributed_time_for_each_server)

    return total_distributed, distributed_time_for_each_server, last_finish_time_for_task, \
           last_finish_time_for_server, server_distributed_for_task, task_chosen, working_order


def get_last_finish_time_for_task(i, m, time_cost, E_trans, last_finish_time_for_task,
                                  last_finish_time_for_server):
    prev_task_finish_time = []
    for prev_task in E_trans[i]:
        prev_task_finish_time.append(last_finish_time_for_task[prev_task])
    earlist_start_time = max(prev_task_finish_time) if prev_task_finish_time else 0
    last_finish_time = np.ones(m) * earlist_start_time
    last_finish_time = np.max(np.stack([last_finish_time, last_finish_time_for_server]), axis=0)
    return last_finish_time + time_cost[i]


def choose_task_in_queue(queue, rest_working_time, candidates=1200):
    """
    启发式的选择下一个工作任务, numpy的函数
    :param queue:
    :param rest_working_time:
    :param candidates:
    :return:
    """
    prob = rest_working_time[queue[:min(candidates, len(queue))]]
    if np.sum(prob) == 0.0:
        return random.choice(range(len(queue)))
    return int(np.where(np.random.multinomial(1, prob / np.sum(prob)) == 1)[0])


def renew_queue(queue, task, rest_working_time):
    cost_task = rest_working_time[task]
    head, tail = 0, len(queue) - 1
    while head < tail:
        mid = (head + tail) // 2
        if mid == head:
            if rest_working_time[queue[head]] < cost_task:
                tail = head
            elif rest_working_time[queue[head]] >= cost_task > rest_working_time[queue[tail]]:
                pass
            else:
                tail += 1
            break
        if rest_working_time[queue[mid]] > cost_task:
            head = mid
        else:
            tail = mid
    queue.insert(tail + 1, task)


def main_loop(params):
    """
    Args
        Nc: 蚁群的循环轮数
        Nant: 蚂蚁数
        T: 节点数
        m: 机器数
    :return:
    """
    Nc = params['Nc']
    Nant = params['Nant']
    T = params['T']
    m = params['m']
    E = params['E']  # out_edges of DAG

    list_batch = params['batch_list']
    available_list = list_batch.pop(0)['table_list']  # 记录不受起批时间限制的表
    mask_available_time_init = np.zeros(T)
    mask_available_time_init[available_list] = 1  # 基于时间的mask, 重要

    E_trans = transpose_G(E)
    in_degs = np.asarray(get_in_deg(E))
    init_pool = np.where((in_degs == 0) * mask_available_time_init)[0].tolist()
    print("init available list ", len(init_pool))

    sigma = params['sigma']
    alpha = params['alpha']
    beta = params['beta']
    rho = params['rho']

    time_cost = params['cost']
    rest_working_time = get_rest_working_time(E, time_cost)
    rest_working_time = np.asarray(rest_working_time)
    init_pool = sorted(init_pool, key=lambda x:rest_working_time[x], reverse=True)  # 对初始任务根据rest_working_hour进行排序

    hormones = 1e-6 * np.ones((T, m))  # hormones is of shape [T, m]

    best_task_distribution_this_round = None
    best_time = float('inf')
    best_working_order = [[] for _ in range(m)]

    for r in range(Nc):
        print(f"round {r}")
        total_time_ants = []

        for _ in tqdm(range(Nant)):
            # 初始化
            task_queue = init_pool[:]  # task_queue 确保是可以安排的任务
            in_degs_copy = in_degs.copy()

            total_distributed = 0  # 总的任务分配时间
            batch_pointer = 0  # 包指针, 用于判断哪些批次可以开始运行
            mask_available_time = mask_available_time_init.copy()
            distributed_time_for_each_server = np.zeros(m)

            last_finish_time_for_task = np.zeros(T)  # 任务的最晚完成时间
            last_finish_time_for_server = np.zeros(m)  # 当前服务器上的最晚完成时间
            server_distributed_for_task = -1 * np.ones(T).astype(np.uint8)  # 任务被分配于哪一台服务器
            working_order = [[] for _ in range(m)]  # 服务器上任务的具体工作顺序

            while task_queue or batch_pointer < len(list_batch):
                while not task_queue and batch_pointer < len(list_batch):
                    # 表明此时起批时间卡住了进程
                    current_time = list_batch[batch_pointer]['start']
                    while batch_pointer < len(list_batch) and list_batch[batch_pointer]['start'] <= current_time:
                        mask_available_time[list_batch[batch_pointer]['table_list']] = 1
                        for ava_task in list_batch[batch_pointer]['table_list']:
                            if in_degs_copy[ava_task] == 0:
                                renew_queue(task_queue, ava_task, rest_working_time)
                        batch_pointer += 1
                    last_finish_time_for_server = current_time * np.ones(m)

                # 从任务池当中随机抽取一个任务, 并扫描是否有新任务可以加入任务池
                task_chosen = choose_task_in_queue(task_queue, rest_working_time)
                task = task_queue.pop(task_chosen)
                for out_vertex in E[task]:
                    in_degs_copy[out_vertex] -= 1
                    if in_degs_copy[out_vertex] == 0 and mask_available_time[out_vertex] == 1:
                        renew_queue(task_queue, out_vertex, rest_working_time)

                # 判断任务task可以被分配于哪些机器上
                time_threshold = total_distributed / m * sigma
                not_available_server_mask = distributed_time_for_each_server > time_threshold
                # 求任务i分配到机器j上的完成时间
                task_last_finish_time_on_server = get_last_finish_time_for_task(task, m, time_cost, E_trans,
                                                                                last_finish_time_for_task,
                                                                                last_finish_time_for_server)

                server_distributed = distribute_task(hormones[task], not_available_server_mask,
                                                     task_last_finish_time_on_server, last_finish_time_for_server,
                                                     alpha=alpha, beta=beta)
                # print(task, 'was distributed to server ', server_distributed)

                total_distributed += time_cost[task]
                distributed_time_for_each_server[server_distributed] += time_cost[task]

                # 确定任务的最终完成时间以及更新对应服务器的最终完成时间
                finish_time = task_last_finish_time_on_server[server_distributed]
                last_finish_time_for_task[task] = finish_time
                last_finish_time_for_server[server_distributed] = finish_time
                server_distributed_for_task[task] = server_distributed
                working_order[server_distributed].append(task)

                current_time = np.min(last_finish_time_for_server)
                while batch_pointer < len(list_batch) and list_batch[batch_pointer]['start'] <= current_time:
                    mask_available_time[list_batch[batch_pointer]['table_list']] = 1
                    for ava_task in list_batch[batch_pointer]['table_list']:
                        if in_degs_copy[ava_task] == 0:
                            renew_queue(task_queue, ava_task, rest_working_time)
                    batch_pointer += 1

            # 一只蚂蚁的安排结束
            # 确定该蚂蚁完成任务的总时间，并更新delta_hormones
            total_time = np.max(last_finish_time_for_server)
            total_time_ants.append(total_time)

            if total_time < best_time:
                best_time = total_time
                best_task_distribution_this_round = server_distributed_for_task
                best_working_order = working_order
                best_finish_time_for_task = last_finish_time_for_task

        dig_to_table_name = params['d2table_name']
        working_order_eng = [[] for _ in range(m)]
        for i, working_order_server in enumerate(best_working_order):
            for task in working_order_server:
                working_order_eng[i].append(dig_to_table_name[task])

        with open(f"../results/working_order_m={m}_iter{r}.txt", 'w') as f:
            for i, order_eng in enumerate(working_order_eng):
                f.write("task flow on server %d :\n" % (i + 1))
                f.write(','.join(order_eng) + '\n')
        np.save(f"../results/finish_time_for_task_m={m}_iter{r}.npy", best_finish_time_for_task)

        # renew hormones
        hormones = hormones * (1 - rho)
        for i in range(T):
            hormones[i, best_task_distribution_this_round[i]] += rho / best_time
        print("min time cost:", min(total_time_ants))
        print("best time", best_time)
        # print(total_time_ants)
        # print(hormones)

    return best_working_order


def test():
    random.seed(600)

    T = 1000
    # 生成一个有80个node的DAG
    E = [[] for _ in range(T)]
    for i in range(0, T - 1):
        # 生成出边
        # print(i)
        # 先看该节点的出边是否有最大节点限制
        set_max_v = random.random() < 0.5
        max_v = T - 1
        if set_max_v:
            max_v = max(min(T - 1, 2 * i), T // 2)

        out_deg = random.randint(2, 5)
        if out_deg >= max_v - i:
            out_deg = random.randint(0, max_v - i)

        for j in range(out_deg):
            while True:
                out_vertex = random.randint(i + 1, max_v)
                if out_vertex not in E[i]:
                    E[i].append(out_vertex)
                    break

    cost = [random.randint(1, 10) for _ in range(T)]
    # print(get_rest_working_time(E, cost))
    # print(E, cost)
    # print(topo_sort(E))
    # print(get_in_deg(G=E))
    # print(transpose_G(E))

    print(E)
    print(cost)
    params = {
        'T': T,
        'm': 10,
        'Nc': 10,
        'Nant': 20,
        'E': E,
        'cost': cost,

        'sigma': 1.2,
        'alpha': 1,
        'beta': 0.5,
        # 'kappa': 20,
        'rho': 0.1
    }
    wo = main_loop(params)
    for o in wo:
        print(len(o))


if __name__ == '__main__':
    # test()

    filename = '../dataset/20210710/cloud20210710_eng.csv'
    batch_start = '../dataset/20210815/batch_start_time.csv'
    G, time_cost, dig_to_table_name, list_batch_start = load_dag(filename, batch_start)
    print(sum(time_cost))

    # const
    m = 120

    params = {
        'T': G.get_num_vertex(),
        'm': m,
        'Nc': 2,
        'Nant': 50,
        'E': G.get_edges(),
        'cost': time_cost,
        'batch_list': list_batch_start,

        'sigma': 1.2,
        'alpha': 0.5,
        'beta': 1.0,
        'rho': 0.1,
        "d2table_name": dig_to_table_name
    }
    working_order = main_loop(params)
    working_order_eng = [[] for _ in range(m)]
    for i, working_order_server in enumerate(working_order):
        for task in working_order_server:
            working_order_eng[i].append(dig_to_table_name[task])

    with open(f"../results/working_order_m={m}.txt", 'w') as f:
        for i, order_eng in enumerate(working_order_eng):
            f.write("task flow on server %d :\n" % (i + 1))
            f.write(','.join(order_eng) + '\n')
