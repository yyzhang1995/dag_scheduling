"""
计算评价指标
"""
from graph_model.building_graph import building_graph
import numpy as np


def get_waiting_time_se():
    filename = '../dataset/20210710/cloud20210710_eng.csv'
    batch_start = '../dataset/20210815/batch_start_time.csv'
    G, time_cost, dig_to_table_name, list_batch_start = building_graph(filename, batch_start)
    table_name_to_dig = {value: key for key, value in dig_to_table_name.items()}

    finish_time = np.load("../results/finish_time_for_task_m=120_iter1.npy")

    num_task = G.get_num_vertex()
    GT = G.transpose()
    se = []
    for i in range(num_task):
        finish_time_i = finish_time[GT.get_out_edge(i)]
        if finish_time_i.size > 0:
            se.append(float(finish_time_i.std()))
    print(sum(se) / len(se))


if __name__ == '__main__':
    get_waiting_time_se()
