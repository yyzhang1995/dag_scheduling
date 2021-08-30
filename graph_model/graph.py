__all__ = ['Graph']


class Graph:
    """
    采用邻接表(出边表)实现的图
    """
    def __init__(self, num_vertex, edges=None, adjacent_table=None):
        """
        通过顶点和边构建图,
        :param num_vertex:
        :param edges: 2-元组, 分别是边的起点和终点
        """
        self._num_vertex = num_vertex
        if edges is None and adjacent_table is None:
            self._edges = [[] for _ in range(num_vertex)]
        if edges is not None:
            self._edges = [[] for _ in range(num_vertex)]
            start, end = edges
            for i, start_v in enumerate(start):
                self._edges[start_v].append(end[i])
        elif adjacent_table is not None:
            self._edges = adjacent_table

    def add_edges(self, start, end):
        if end not in self._edges[start]:
            self._edges[start].append(end)

    def get_edges(self):
        return self._edges

    def get_out_edge(self, i):
        assert i < self._num_vertex
        return self._edges[i]

    def get_num_vertex(self):
        return self._num_vertex

    def transpose(self):
        """
        图转置
        :return:
        """
        trans_edges = [[] for _ in range(self._num_vertex)]
        for start_vertex, out_edges in enumerate(self._edges):
            for end_vertex in out_edges:
                trans_edges[end_vertex].append(start_vertex)
        return Graph(self._num_vertex, adjacent_table=trans_edges)

    def topo_sort(self):
        """
        返回图G的一个拓扑排序, 如果G不是DAG则报错
        :return:
        """
        in_vertex = [0] * self._num_vertex
        for out_edges in self._edges:
            for out_vertices in out_edges:
                in_vertex[out_vertices] += 1
        queue = []
        for i in range(self._num_vertex):
            if in_vertex[i] == 0:
                queue.append(i)
        topo = []
        while queue:
            v = queue.pop()
            for out_vertex in self._edges[v]:
                in_vertex[out_vertex] -= 1
                if in_vertex[out_vertex] == 0:
                    queue.append(out_vertex)
            topo.append(v)
        if len(topo) < self._num_vertex:
            raise TypeError("Not a DAG")
        return topo

    def degree_out_vertices(self):
        degree = []
        for i in range(self._num_vertex):
            degree.append(len(self._edges[i]))
        return degree


def test():
    num_vertex = 7
    edges = ([0, 0, 1, 2, 4, 5], [1, 4, 2, 3, 5, 6])
    G = Graph(num_vertex, edges)
    print(G.get_edges())
    print(G.transpose().get_edges())
    print(G.topo_sort())


if __name__ == '__main__':
    test()
