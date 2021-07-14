import numpy as np
import random
import torch


def load_dag():
    """
    读取DAG图
    :return:
    """
    pass


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


def distribute_task(hormone_i, mask, alpha):
    candidate_tau = torch.from_numpy(hormone_i.copy())
    candidate_tau[mask] = 0.0
    # if np.all(mask):
    #     return random.randint(0, len(mask) - 1)
    if torch.all(candidate_tau == 0.0):
        return torch.multinomial(torch.from_numpy(1 - mask).float(), 1).item()
    return torch.multinomial(candidate_tau ** alpha, 1).item()


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
    E = params['E'] # out_edges of DAG
    E_trans = transpose_G(E)
    in_degs = np.asarray(get_in_deg(E))
    init_pool = np.where(in_degs == 0)[0].tolist()

    sigma = params['sigma']
    alpha = params['alpha']
    kappa = params['kappa'] # delta_hormones(i, j, k) = kappa / Zk
    rho = params['rho']

    time_cost = params['cost']

    hormones = 5 * np.ones((T, m)) # hormones is of shape [T, m]

    for r in range(Nc):
        print(f"round {r}")
        delta_hormones = np.zeros_like(hormones)
        total_time_ants = []

        for ant in range(Nant):
            # print(f"ant {ant}")
            total_distributed = 0
            distributed_time_for_each_server = np.zeros(m)

            task_queue = init_pool[:]
            in_degs_copy = in_degs.copy()

            last_finish_time_for_task = np.zeros(T)
            last_finish_time_for_server = np.zeros(m)
            server_distributed_for_task = -1 * np.ones(T).astype(np.uint8)
            while task_queue:
                # 从任务池当中随机抽取一个任务, 并扫描是否有新任务可以加入任务池
                task = task_queue.pop(random.randint(0, len(task_queue) - 1))
                for out_vertex in E[task]:
                    in_degs_copy[out_vertex] -= 1
                    if in_degs_copy[out_vertex] == 0:
                        task_queue.append(out_vertex)

                # 判断任务task可以被分配于哪些机器上
                time_threshold = total_distributed / m * sigma
                not_available_server_mask = distributed_time_for_each_server > time_threshold
                server_distributed = distribute_task(hormones[task], not_available_server_mask, alpha=alpha)
                # print(task, 'was distributed to server ', server_distributed)

                total_distributed += time_cost[task]
                distributed_time_for_each_server[server_distributed] += time_cost[task]

                # 确定任务的最终完成时间以及更新对应服务器的最终完成时间
                try:
                    other_finish_time = np.max(last_finish_time_for_task[E_trans[task]])
                except ValueError:
                    other_finish_time = 0
                this_finish_time = last_finish_time_for_server[server_distributed]
                finish_time = max(this_finish_time, other_finish_time) + time_cost[task]
                last_finish_time_for_task[task] = finish_time
                last_finish_time_for_server[server_distributed] = finish_time
                server_distributed_for_task[task] = server_distributed

            # 确定该蚂蚁完成任务的总时间，并更新delta_hormones
            # print(server_distributed_for_task)
            # print(last_finish_time_for_task)
            total_time = np.max(last_finish_time_for_server)
            total_time_ants.append(total_time)
            for i in range(T):
                delta_hormones[i, server_distributed_for_task[i]] += kappa / total_time
            # print(delta_hormones)

        # renew hormones
        hormones = hormones * (1 - rho) + delta_hormones
        print("average time cost:", sum(total_time_ants) / Nant)
        print(total_time_ants)


def test():
    random.seed(600)

    T = 10
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
    # print(E, cost)
    # print(topo_sort(E))
    # print(get_in_deg(G=E))
    # print(transpose_G(E))

    print(E)
    print(cost)
    params = {
        'T': T,
        'm': 3,
        'Nc': 100,
        'Nant': 5,
        'E': E,
        'cost': cost,

        'sigma': 1.1,
        'alpha': 1,
        'kappa': 200,
        'rho': 0.1
    }
    main_loop(params)


if __name__ == '__main__':
    test()



