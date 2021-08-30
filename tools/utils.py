from graph_model.building_graph import building_graph

def calculate_time(working_order):
    filename = '../dataset/20200710/cloud20210710_eng.csv'
    batch_start = '../dataset/20200710/batch_start_time.csv'
    G, time_cost, dig_to_table_name, list_batch_start = building_graph(filename, batch_start)

    table_name_to_dig = {value: key for key, value in dig_to_table_name.items()}
    num_tables = G.num_vertex()

    list_start_time = [0 for _ in range(num_tables)]
    for bs in list_batch_start:
        earliest_time = bs['time']
        for tb_name in bs['table']:
            list_start_time[tb_name] = earliest_time

    task_end_time = [None for _ in range(num_tables)]

    m = len(working_order)
    m_pt = [-1 for _ in range(m)]
    latest_m_time = [0 for _ in range(m)]
    processed_task = 0
    while processed_task < num_tables:
        for i in range(m):
            cur_task = working_order[i][m_pt[i]] + 1
            dep_task = [latest_m_time[i]]
            for t in G.get_out_edge(cur_task):
                dep_task.append(task_end_time[t])
            if None in dep_task:
                continue
            task_end_time[cur_task] = max(dep_task) + time_cost[cur_task]
            m_pt[i] += 1
            latest_m_time[i] = task_end_time[cur_task]
