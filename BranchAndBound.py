import logging
import copy
import time
import random
from tqdm import tqdm
import numpy as np


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%H:%M:%S')

class BranchAndBound:
    def __init__(self, lb=0):
        self.lb = lb
        
        self.best_clique = []
        self.cur_max = 0

    def Clique(self, graph, U, size, cur_clique):
        if len(U) == 0:
            if size > self.cur_max:
                self.cur_max = size
                self.best_clique = cur_clique
            return

        while len(U) > 0:
            if size + len(U) <= self.cur_max: # pruning 4
                return

            vertex = U.pop()
            new_cur_clique = cur_clique[:]
            new_cur_clique.append(vertex)
            
            # pruning 5
            neib_vertex = set(v for v in graph[vertex] if len(graph[v]) >= self.cur_max)
            new_U = U.intersection(neib_vertex)

            self.Clique(graph, new_U, size + 1, new_cur_clique)

    def MaxClique(self, graph, lb=0):
        # nodes are labeled as 1, 2, ....no_of_vertices
        no_of_vertices = len(graph)
        self.cur_max = lb
        self.best_clique = []

        for i in tqdm(range(1, no_of_vertices + 1), desc='Running bnb loops'):
            if str(i) not in graph:
                continue
            neib_vi = graph[str(i)]

            # pruning 1
            if len(neib_vi) >= self.cur_max: 
                U = set()
                cur_clique = [str(i)]

                for j in neib_vi:
                    # pruning 2
                    if int(j) > i: 
                        # pruning 3
                        if len(graph[j]) >= self.cur_max: 
                            U.add(j)

                self.Clique(graph, U, 1, cur_clique)


    def run(self, graph):
        start_time = time.time()
        self.MaxClique(graph, self.lb)
        end_time = time.time()
        
        s = self.cur_max
        t = (end_time - start_time) * 1000

        logging.info(f"clique size: {s}, time(ms): {t:.3f}")
        return (s, t)