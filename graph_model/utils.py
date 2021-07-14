import re
from graph import Graph

__all__ = ['topo_sort', 'change_time_to_min', 'num_dat_without_intersection']


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


def num_dat_without_intersection(G: Graph):
    num_vertex = G.get_num_vertex()

    in_degree = [0] * num_vertex
    for i in range(num_vertex):
        for out_vertex in G.get_out_edge(i):
            in_degree[out_vertex] += 1

    start_vertices = []
    for i in range(num_vertex):
        if in_degree[i] == 0:
            start_vertices.append(i)

    num_no_intersection = 0
    color = [-1] * num_vertex
    for start_vertex in start_vertices:
        queue = [start_vertex]
        flag = True
        # bfs
        while queue:
            v_in = queue.pop()
            for v_out in G.get_out_edge(v_in):
                if color[v_out] == -1:
                    color[v_out] = start_vertex
                    queue.append(v_out)
                elif color[v_out] == start_vertex:
                    pass
                else:
                    flag = False
        if flag: num_no_intersection += 1
    return num_no_intersection


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



if __name__ == '__main__':
    G = [[2], [0], []]
    print(topo_sort(G))
