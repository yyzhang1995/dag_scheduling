import matplotlib.pyplot as plt
from graph_model.building_graph import building_graph
import numpy as np


def vision():
    filename = '../dataset/20210710/cloud20210710_eng.csv'
    batch_start = '../dataset/20210815/batch_start_time.csv'
    G, time_cost, dig_to_table_name, list_batch_start = building_graph(filename, batch_start)
    table_name_to_dig = {value: key for key, value in dig_to_table_name.items()}
    m = 120
    m_show = 120

    working_order_file = "../results/working_order_m=120_iter1.txt"
    finish_time = np.load("../results/finish_time_for_task_m=120_iter1.npy")
    max_time = np.max(finish_time)
    with open(working_order_file, 'r') as f:
        working_order = [l.strip().split(',') for i, l in enumerate(f.readlines()) if i % 2 == 1]
    print(working_order[0])

    for i in range(m_show):
        plt.subplot(m_show + 1, 1, i + 1)
        plt.xlim(0, max_time)
        plt.ylim(0, 2)

        working_order_i = [table_name_to_dig[tn] for tn in working_order[i]]
        for task in working_order_i:
            plt.plot([finish_time[task] - time_cost[task], finish_time[task]], [1, 1])
        for bs_info in list_batch_start:
            if not bs_info['table_list']: continue
            plt.plot([bs_info['start'], bs_info['start']], [0, 2], 'k', linestyle='dashed', linewidth=0.3)

        plt.axis('off')

    plt.subplot(m_show + 1, 1, m_show + 1)
    plt.axis('off')
    plt.xlim(0, max_time)
    plt.ylim(0, 10)
    for i in range(0, int(max_time), 50):
        plt.annotate(str(i), xy=(i, 0), size=3)

    plt.show()

if __name__ == '__main__':
    vision()
    # plt.plot([np.asarray(5), np.asarray(10)], [1, 1], 'k-', linestyle='dashed', linewidth=0.3)
    #
    # plt.axis('off')
    # plt.annotate("hello1", xy=(8, 1), size=5)
    # plt.annotate("hello2", xy=[10, 1])
    # plt.show()
