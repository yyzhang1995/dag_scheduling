import re
from graph_model.graph import Graph
from collections import defaultdict

__all__ = ['topo_sort', 'change_time_to_min', 'num_dag_without_intersection', 'get_rest_working_time', 'get_system']


def topo_sort(G):
    """
    G以邻接表的形式（出边）给出
    :param G: list
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


def num_dag_without_intersection(G: Graph):
    """
    统计不连通的DAG数目
    :param G:
    :return:
    """
    num_vertex = G.get_num_vertex()

    in_degree = [0] * num_vertex
    for i in range(num_vertex):
        for out_vertex in G.get_out_edge(i):
            in_degree[out_vertex] += 1

    start_vertices = []
    for i in range(num_vertex):
        if in_degree[i] == 0:
            start_vertices.append(i)
    print("number of start vertex:", len(start_vertices))
    parent_start_vertices = {vertex: None for vertex in start_vertices}
    num_node_start_vertices = {vertex: 1 for vertex in start_vertices}

    num_no_intersection = 0
    color = [-1] * num_vertex
    for start_vertex in start_vertices:
        queue = [start_vertex]
        color[start_vertex] = start_vertex
        flag = True
        # bfs
        while queue:
            v_in = queue.pop()
            for v_out in G.get_out_edge(v_in):
                if color[v_out] == -1:
                    color[v_out] = start_vertex
                    num_node_start_vertices[start_vertex] += 1
                    queue.append(v_out)
                elif color[v_out] == start_vertex:
                    pass
                else:
                    if parent_start_vertices[color[v_out]] is None:
                        parent_start_vertices[start_vertex] = color[v_out]
                    else:
                        parent_start_vertices[start_vertex] = parent_start_vertices[color[v_out]]
                    flag = False
        if flag: num_no_intersection += 1

    # print(any([c == -1 for c in color]))
    print("sum of num_node_start_vertices", sum(num_node_start_vertices.values()))
    num_of_node_in_dag = defaultdict(int)
    for start_vertex, num_node in num_node_start_vertices.items():
        if parent_start_vertices[start_vertex] is None:
            num_of_node_in_dag[start_vertex] += num_node
        else:
            num_of_node_in_dag[parent_start_vertices[start_vertex]] += num_node
    return num_no_intersection, [num_node for num_node in num_of_node_in_dag.values()]


def change_time_to_min(time):
    """
    time in the form of year/month/day hh:mm or hh:mm:ss
    :param time:
    :return:
    """
    if type(time) != str: print(time)
    if '/' in time:
        t = re.findall('\d{1,2}:\d{1,2}', time)[0]
    else:
        t = re.findall('\d{1,2}:\d{1,2}', time)[0]
    hh, mm = t.split(":")
    return int(hh) * 60 + int(mm)


def get_rest_working_time(G, time_cost):
    if isinstance(G, list):
        G = Graph(len(G), adjacent_table=G)

    G_trans = G.transpose()
    longest_rest_work_time = {}
    out_vertices_degree = []
    queue = []
    for i in range(G.get_num_vertex()):
        out = len(G.get_out_edge(i))
        out_vertices_degree.append(out)
        if out == 0:
            queue.append(i)
    while queue:
        vertex = queue.pop()
        rest = 0
        for out_vertex in G.get_out_edge(vertex):
            rest = max(rest, longest_rest_work_time[out_vertex])
        longest_rest_work_time[vertex] = time_cost[vertex] + rest
        for in_vertex in G_trans.get_out_edge(vertex):
            out_vertices_degree[in_vertex] -= 1
            if out_vertices_degree[in_vertex] == 0:
                queue.append(in_vertex)
    # print(longest_rest_work_time)
    return [longest_rest_work_time[i] for i in range(G.get_num_vertex())]


def get_system(schema_name):
    schema, name = schema_name.split('@')
    if schema.endswith('_f'):
        return name.split('_')[1]
    else:
        return ''


if __name__ == '__main__':
    G = Graph(8, adjacent_table = [[1, 2], [], [4], [2], [], [4], [7], []])
    print(num_dag_without_intersection(G))
