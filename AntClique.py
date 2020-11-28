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

class AntClique:
    def __init__(self, num_ants=7, taomin=0.01, taomax=4, alpha=2, rho=.995, max_cycles=3000):
        self.num_ants = num_ants
        self.taomin = taomin
        self.taomax = taomax
        self.alpha = alpha
        self.rho = rho
        self.max_cycles = max_cycles
        
        self.graph = None
        self.best_clique_info = None
        
    def initialize_pheromone_trails(self, graph):
        self.graph = copy.deepcopy(graph)
        self.best_clique_info = {
            'clique': set(),
            'req_time': -1,
            'req_cycles': -1,
        } 

        # initialize all edges with taomax pheromone
        for n, nbrs in graph.items():
            for nbr, attrs in nbrs.items():
                attrs['pheromone'] = self.taomax
                self.graph[n][nbr] = attrs
    

    def construct_clique(self, ant_idx):
        clique = set()
        candidates = set()
        neigbors = lambda node: set(self.graph[node].keys())
        pheromone_factor = lambda node: sum(self.graph[node][clique_node]['pheromone'] for clique_node in clique)  
        
        initial_vertex = random.sample(self.graph.keys(), 1)[0]
        clique.add(initial_vertex)
        candidates.update(neigbors(initial_vertex))

        while candidates:
            pheromone_factors = [pheromone_factor(node) ** self.alpha for node in candidates]
            pheromone_probs = [factor / sum(pheromone_factors) for factor in pheromone_factors]
    
            selected_vertex = np.random.choice(list(candidates), size=1, p=pheromone_probs)[0]

            clique.add(selected_vertex)
            candidates = candidates.intersection(neigbors(selected_vertex))

        return (ant_idx, clique)

    def update_pheromone_trails(self, iteration_no, start_time, cliques):
        best_ant_idx, best_clique = max(cliques, key=lambda t: len(t[1]))
        
        # update global info
        if len(best_clique) > len(self.best_clique_info['clique']):
            self.best_clique_info = {
                'clique': best_clique,
                'req_time': (time.time() - start_time) * 1000,
                'req_cycles': iteration_no + 1
            }

        c_best = len(self.best_clique_info['clique'])
        c_k = len(best_clique)

        # evaporate pheromone on all edges
        for n, nbrs in self.graph.items():
            for nbr, attrs in nbrs.items():
                attrs['pheromone'] = max(self.taomin, self.rho * attrs['pheromone'])
                self.graph[n][nbr] = attrs

        # deposit pheromone for best ant
        for n in best_clique:
            for nbr in best_clique:
                if n == nbr:
                    continue
                
                attrs = self.graph[n][nbr]
                attrs['pheromone'] = min(self.taomax, (1 / (1 + c_best - c_k)) + attrs['pheromone'])
                self.graph[n][nbr] = attrs


    def run(self, graph, use_threading=False):
        start_time = time.time()
        
        self.initialize_pheromone_trails(graph)

        for iteration_no in tqdm(range(self.max_cycles), desc='Running ant-clique loops'):
            if use_threading:
                with Pool(self.num_ants) as pool:
                    formed_cliques = pool.map(self.construct_clique, range(self.num_ants))
            else:
                formed_cliques = [self.construct_clique(i) for i in range(self.num_ants)]

            self.update_pheromone_trails(iteration_no, start_time, formed_cliques)
        
        s = len(self.best_clique_info['clique'])
        t = self.best_clique_info['req_time']
        c = self.best_clique_info['req_cycles']

        logging.info(f"clique size: {s}, req cycles: {c}, req time: {t:.3f}")
        return (s, t, c)



