# import sys
# sys.path.append(__file__)
import pandas as pd
import numpy as np
from graph_model.utils import *
from graph_model.graph import Graph
import matplotlib.pyplot as plt


__all__ = ['building_graph']

pd.options.display.max_columns = None
# 需要将文件当中的列名称分别改为
# 'table_schema', 'table_name', 'start', 'end', 'dep_tab_schema', 'dep_table_name', 'dep_start', 'dep_end'


def building_graph(filename, batch_start):
    """
    输入文件, 输出DAG图和任务花费(dict)
    return
    {
        G: Graph
        time_cost: list
    }
    """
    features = ['table_schema', 'table_name', 'start', 'end', 'dep_table_schema', 'dep_table_name', 'dep_start',
                'dep_end']
    # 不区分大小写, 统一转化为小写
    df = pd.read_csv(filename, encoding='utf8')[features]
    bs = pd.read_csv(batch_start)[['system', 'batch_start_time']]
    df[['table_schema', 'table_name', 'dep_table_schema', 'dep_table_name']] = \
        df[['table_schema', 'table_name', 'dep_table_schema', 'dep_table_name']].apply(
            lambda x: x.astype(str).str.lower())
    num_sample = df.shape[0]
    print("number of edges:", num_sample)

    # 处理起批时间
    bs['batch_start_time'] = bs['batch_start_time'].map(change_time_to_min)
    bs = bs.drop(bs[bs['batch_start_time'] >= 600].index)
    bs[['system']] = bs[['system']].apply(lambda x: x.astype(str).str.lower())
    print("batch start time:", bs.shape)

    # 去重并获得所有表时间信息
    table_info = df[features[:4]].reset_index(drop=True)
    dep_table_info = df[features[4:]].reset_index(drop=True)
    dep_table_info.columns = features[:4]
    table_info = pd.concat((table_info, dep_table_info))
    table_info = table_info.drop_duplicates(subset=['table_schema', 'table_name']).reset_index(drop=True)
    table_info = table_info.fillna("0:0:0")  # 如果没有给出时间，则统一认为该任务在最开始已经完成
    table_info['start'] = table_info['start'].map(change_time_to_min)
    table_info['end'] = table_info['end'].map(change_time_to_min)
    table_info['interval'] = table_info['end'] - table_info['start']
    num_vertex = table_info.shape[0]
    # (30359, 5)
    # 转字典
    table_info['order'] = table_info.index
    index_name = [f'{tab_schema}@{name}' for tab_schema, name in
                  zip(table_info['table_schema'], table_info['table_name'])]
    table_info['index'] = index_name
    table_info['system'] = table_info['index'].map(get_system)

    # 合并表和起批时间
    table_info = table_info.merge(bs, on='system', how='left').fillna(0)
    table_info = table_info.set_index(['index'])

    # 观察表的处理情况
    print("shape of table info: ", table_info.shape)
    print(table_info.head())

    # 生成图, 并检验是否是DAG
    eng = [f'{tab_schema}@{name}' for tab_schema, name in zip(df['table_schema'], df['table_name'])]
    dep_eng = [f'{tab_schema}@{name}' for tab_schema, name in zip(df['dep_table_schema'], df['dep_table_name'])]

    name_dict = {name: i for name, i in zip(table_info.index, table_info['order'])}  # name: order
    dig_to_eng = {value: key for key, value in name_dict.items()}  # order: name

    adjacent = [[] for _ in range(num_vertex)]
    eng_dig_before = [name_dict[name] for name in eng]
    dep_eng_dig_before = [name_dict[name] for name in dep_eng]

    # 去除时间相悖的边
    dep_eng_dig, eng_dig = [], []
    for i in range(num_sample - 1, -1, -1):
        dep_table_name = dig_to_eng[dep_eng_dig_before[i]]
        table_name = dig_to_eng[eng_dig_before[i]]
        if table_info.loc[dep_table_name]['end'] <= table_info.loc[table_name]['start']:
            dep_eng_dig.append(dep_eng_dig_before[i])
            eng_dig.append(eng_dig_before[i])
    print("edges after remove contradictory pairs: ", len(eng_dig))

    for i in range(len(dep_eng_dig)):
        if eng_dig[i] != dep_eng_dig[i]:
            adjacent[dep_eng_dig[i]].append(eng_dig[i])
    G = Graph(num_vertex, edges=(dep_eng_dig, eng_dig))
    G.topo_sort()  # 如果不是DAG, 会报错

    # 生成任务消耗时间list
    time_cost = []
    for i in range(num_vertex):
        time_cost.append(table_info.loc[dig_to_eng[i]]['interval'])

    # 获得起批时间与表的关系
    list_of_batch = [{'name': sys_name, 'start': sys_start,
                      "table_list": table_info[table_info['system'] == sys_name]['order'].tolist()}
                     for sys_name, sys_start in zip(bs['system'], bs['batch_start_time'])]
    list_of_batch.sort(key=lambda x: x['start'])
    print(list_of_batch)
    list_of_batch.insert(0, {'name': "", "start": 0,
                             "table_list": table_info[table_info['batch_start_time'] == 0]['order'].tolist()})
    # print(sum([len(item['table_list']) for item in list_of_batch]))
    return G, time_cost, dig_to_eng, list_of_batch


def view_data():
    filename = '../dataset/20210710/cloud20210710_eng.csv'
    batch_start = '../dataset/20210815/batch_start_time.csv'
    G, time_cost, _, _ = building_graph(filename, batch_start)
    # G1, _, _ = building_graph(filename)
    # G_out = G.degree_out_vertices()
    # G_out.sort()
    # G1_out = G1.degree_out_vertices()
    # G1_out.sort()
    # for i in range(len(G_out)):
    #     if G1_out[i] != G_out[i]:
    #         print('not the same')
    #     break

    num_dags, dag_nodes = num_dag_without_intersection(G)
    print("number of independent dag", num_dags)
    # print(num_dag_without_intersection(G1)[0])
    # print(num_dag_without_intersection(G)[0])
    # print(num_dag_without_intersection(G)[0])

    # for s in dag_nodes:
    #     if s >= 100: dag_nodes.remove(s)

    from collections import defaultdict
    numbers = defaultdict(int)
    for num_node in dag_nodes:
        numbers[num_node] += 1
    for key in sorted(numbers.keys()):
        print(key, numbers[key])

    # plt.hist([num for num in dag_nodes if num <= 50])
    # plt.show()

    # import networkx as nx
    # G_networkx = nx.DiGraph()
    # G_networkx.add_nodes_from(range(G.get_num_vertex()))
    # for i in range(G.get_num_vertex()):
    #     for out_v in G.get_out_edge(i):
    #         G_networkx.add_edge(i, out_v)
    # nx.draw_networkx()
    # nx.draw(G_networkx)
    # plt.show()

    # 观察time_cost的分布
    plt.hist([c for c in time_cost if c <= 20])
    plt.show()


if __name__ == '__main__':
    view_data()

    # filename = '../dataset/cloud20210710_eng.csv'
    # G, time_cost = building_graph(filename)
    # print(time_cost[:20])