import pandas as pd
import numpy as np
from utils import *
from graph import Graph


__all__ = ['building_graph']

pd.options.display.max_columns = None
# 需要将文件当中的列名称分别改为
# 'table_schema', 'table_name', 'start', 'end', 'dep_tab_schema', 'dep_table_name', 'dep_start', 'dep_end'

def building_graph(filename):
    """
    输入文件, 输出DAG图和任务花费(dict)
    return
    {
        G: Graph
        time_cost: list
    }
    """
    features = ['table_schema', 'table_name', 'start', 'end', 'dep_tab_schema', 'dep_table_name', 'dep_start',
                'dep_end']
    # 不区分大小写, 统一转化为小写
    df = pd.read_csv(filename, encoding='utf8')[features]
    df[['table_schema', 'table_name', 'dep_tab_schema', 'dep_table_name']] = \
        df[['table_schema', 'table_name', 'dep_tab_schema', 'dep_table_name']].apply(
            lambda x: x.astype(str).str.lower())
    num_sample = df.shape[0]
    print("number of edges:", num_sample)

    # 去重并获得所有表时间信息
    table_info = df[features[:4]].reset_index(drop=True)
    dep_table_info = df[features[4:]].reset_index(drop=True)
    dep_table_info.columns = features[:4]
    table_info = pd.concat((table_info, dep_table_info))
    table_info = table_info.drop_duplicates(subset=['table_schema', 'table_name'])
    table_info = table_info.fillna("0:0:0")  # 如果没有给出时间，则统一认为该任务在最开始已经完成
    table_info['start'] = table_info['start'].map(change_time_to_min)
    table_info['end'] = table_info['end'].map(change_time_to_min)
    table_info['interval'] = table_info['end'] - table_info['start']
    print("shape of table info: ", table_info.shape)
    num_vertex = table_info.shape[0]
    # (33572, 4)
    # 转字典
    index_name = [f'{tab_schema}_{name}' for tab_schema, name in
                  zip(table_info['table_schema'], table_info['table_name'])]
    table_info['index'] = index_name
    table_info = table_info.set_index(['index'])
    print(table_info.head())

    # 生成图, 并检验是否是DAG
    eng = [f'{tab_schema}_{name}' for tab_schema, name in zip(df['table_schema'], df['table_name'])]
    dep_eng = [f'{tab_schema}_{name}' for tab_schema, name in zip(df['dep_tab_schema'], df['dep_table_name'])]

    name_unique = list(set(eng + dep_eng))

    name_dict = {name: i for i, name in enumerate(name_unique)}
    dig_to_eng = {value: key for key, value in name_dict.items()}

    adjacent = [[] for _ in range(len(name_unique))]
    eng_dig = [name_dict[name] for name in eng]
    dep_eng_dig = [name_dict[name] for name in dep_eng]

    # 去除时间相悖的边
    for i in range(num_sample - 1, -1, -1):
        dep_table_name = dig_to_eng[dep_eng_dig[i]]
        table_name = dig_to_eng[eng_dig[i]]
        if table_info.loc[dep_table_name]['end'] > table_info.loc[table_name]['start']:
            dep_eng_dig.pop(i)
            eng_dig.pop(i)
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

    return G, time_cost


def view_data():
    filename = '../dataset/cloud20210710_eng.csv'
    G, time_cost = building_graph(filename)
    print(num_dat_without_intersection(G))


if __name__ == '__main__':
    view_data()

    # filename = '../dataset/cloud20210710_eng.csv'
    # G, time_cost = building_graph(filename)
    # print(time_cost[:20])