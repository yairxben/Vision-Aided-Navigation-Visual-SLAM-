import numpy as np
import heapq
from Edge import Edge

COV_DIM = 6


class VertexGraph:

    def __init__(self, vertices_num, rel_covs=None, weighted_det=True):
        """
        :param rel_covs: Relative covariances between consecutive cameras
        :param directed: If True, the graph is directed; otherwise, it is undirected.
        """
        self.__v_num = vertices_num
        self.__rel_covs = rel_covs
        self.create_vertex_graph()
        self.__edges = dict()

    def create_vertex_graph(self):
        """
        Creates the vertex graph as an adjacency list
        """
        # self.__graph = {i: [] for i in range(self.__v_num)}
        # for i in range(self.__v_num - 1):
        #     edge = Edge(i, i + 1, self.__rel_covs[i])
        #     self.__graph[i].append((i + 1, edge._Edge__weight))  # append tuple (target, weight)
        #
        self.__graph = {i: {} for i in range(self.__v_num)}
        self.__edges = dict()
        for i in range(self.__v_num - 1):
            edge = Edge(i, i + 1, self.__rel_covs[i])
            self.__graph[i][i + 1] = edge  # append to the i-th edges the edge of i + 1 (as hashmap key- i+1->edge)
            self.__edges[(i, i + 1)] = edge

    def find_shortest_path(self, source, target):
        """

        Args:
            source:
            target:

        Returns:

        """
        # use min heap for dists
        dists = [float('inf')] * self.__v_num

        parents = [-1] * self.__v_num
        calculated_vertices = [False] * self.__v_num
        dists[source] = 0
        min_heap = [(0, source)]  # (distance, node)

        while min_heap:
            curr_dist, u = heapq.heappop(min_heap)

            if calculated_vertices[u]:
                continue
            calculated_vertices[u] = True

            for neighbor, weight in self.__graph.get(u, {}).items():
                if not calculated_vertices[neighbor] and dists[neighbor] > curr_dist + weight.get_weight():
                    dists[neighbor] = curr_dist + weight.get_weight()
                    parents[neighbor] = u
                    heapq.heappush(min_heap, (dists[neighbor], neighbor))

        # Build the path from source to target (if needed)
        path = []
        if dists[target] != float('inf'):
            v = target
            while v != -1:
                path.append(v)
                v = parents[v]
            path.reverse()
        return dists[target], path

    def estimate_rel_cov(self, path):
        """
        Compute the estimated relative covariance between to cameras in the path the connecting them
        :param path: list of cameras indexes where the first index contains the first camera and the last index contains
         the last camera in the path
         :return estimated covariance
        """
        estimated_rel_cov = np.zeros((COV_DIM, COV_DIM))
        for i in range(1, len(path)):  # don't include first rel_covs at the path
            edge = self.get_edge_between_vertices(path[1][i - 1], path[1][i])
            estimated_rel_cov += edge.get_cov()
        return estimated_rel_cov

    def get_edge_between_vertices(self, source, target):
        """
        Returns the edge between first_v and target
        """
        return self.__graph[source][target]

    def add_edge(self, prev_ind, key_frame_ind, cov_mat):
        """
        Adds an edge to the graph
        :param edge: Edge object
        """
        edge = Edge(prev_ind, key_frame_ind, cov_mat)
         # add to the graph
        self.__graph[prev_ind][key_frame_ind] = edge
        self.__edges[(prev_ind, key_frame_ind)] = edge
