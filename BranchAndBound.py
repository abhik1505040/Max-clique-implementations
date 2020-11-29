import logging
import copy
import time
import random
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

random.seed(0)
np.random.seed(0)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%H:%M:%S')

class BranchAndBound:
    def __init__(self, lb=0):
        self.lb = lb
        
        self.graph = None
        self.best_clique = None

    def init_graph(self, graph):
        self.graph = copy.deepcopy(graph)
        self.best_clique = []

    def Clique(self, graph, U, size, cur_clique):
        if len(U) == 0:
            if size > self.cur_max:
                self.cur_max = size
                self.best_clique = copy.deepcopy(cur_clique)
            return "Done"
        while len(U) > 0:
            if size + len(U) <= self.cur_max:
                return "Done"
            vertex = list(U)[0]
            U.remove(list(U)[0])
            new_cur_clique = copy.deepcopy(cur_clique)
            new_cur_clique.append(vertex)
            neib_vertex_temp = list(graph[vertex].keys())
            neib_vertex = []
            for u in range(len(neib_vertex_temp)):
                degree_u = len(list(graph[neib_vertex_temp[u]].keys()))
                if degree_u >= self.cur_max:
                    neib_vertex.append(neib_vertex_temp[u])
            new_U = U.intersection(set(neib_vertex))
            self.Clique(graph, new_U, size+1, new_cur_clique)

        return "Done"

            

    def MaxClique(self, graph, lb):
        no_of_vertices = len(list(graph))
        self.cur_max = lb
        for i in range(no_of_vertices):
            neib_vi = list(graph[str(i+1)].keys())
            if len(neib_vi) >= self.cur_max:
                U = set()
                cur_clique = [str(i+1)]
                for j in range(len(neib_vi)):
                    _i = int(i+1)
                    _j = int(neib_vi[j])
                    if _j > _i:
                        if len(list(graph[str(_j)].keys())) >= self.cur_max:
                            U.add(str(_j))
                self.Clique(graph, U, 1, cur_clique)
        return "Done"


    def run(self, graph, use_threading=False):
        start_time = time.time()
        
        self.init_graph(graph)

        self.MaxClique(graph, self.lb)

        end_time = time.time()
        s = self.cur_max
        t = (end_time - start_time)

        logging.info(f"clique size: {s}, time(sec): {t:.3f}")
        return (s, t)