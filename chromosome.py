"""
=============================================================================
chromosome.py  —  Clase Chromosome original del Dr. Johan Carvajal Godínez
=============================================================================
Este archivo NO se modifica. Se usa tal cual para garantizar que QEA y GA
resuelvan exactamente el mismo problema con las mismas restricciones.
=============================================================================
"""

import random as rd
import numpy as np
import networkx as nx


class Chromosome:   # Class For Creating Topologies for MAS-based Architectures
    def __init__(self, n, cost_par_mat):
        self._genes = []
        self._golden_genes = []
        self._graph = nx.Graph(name="MAS-ARCH")
        self._valid = 0
        self._fitness = 0
        self._cost_par = cost_par_mat
        self._target_cost = 0
        self._cost_tot = 0
        self._A = np.zeros((n, n))
        self._num_agents = n
        self._adj_list = []
        self._master_nodes = []
        self._functional_nodes = []
        self._team_edges = []
        self._hierarchy_edges = []
        self._int_constraints = []
        self._fixed_cost = 0
        self._coherence = False
        self._degree_coord = False
        self._degree_funct = False
        self._clustering_array = [n-1]
        self._degree_range = [3, (round((n-1)/3)+2)]
        i = 0
        while i < (self._num_agents*(self._num_agents-1)/2):
            if i < (self._num_agents-1):
                self._genes.append(1)
                self._golden_genes.append(1)
            else:
                if rd.random() >= 0.5:
                    self._genes.append(1)
                    self._golden_genes.append(0)
                else:
                    self._genes.append(0)
                    self._golden_genes.append(0)
            i += 1

    def is_coordinator(self, node):
        return node in self.get_master_nodes()

    def get_coherence(self):
        self._coherence = False
        degrees = list(self.get_graph().degree().values())
        if len(degrees) > 1:
            degrees.pop(0)
        if len(degrees) > 0:
            if min(degrees) >= self._degree_range[0]:
                if max(degrees) <= self._degree_range[1]:
                    self._coherence = True
                else:
                    self._coherence = False
        return self._coherence

    def get_degree_range(self):
        return self._degree_range

    def set_gen(self, gen):
        self._genes[gen] = 1

    def get_target_cost(self):
        self._target_cost = round(self.get_fixed_cost()+(((self._num_agents*(self._num_agents-1)/2)-self.get_fixed_cost())/2))
        return self._target_cost

    def get_cost_par_mat(self):
        return self._cost_par

    def get_adj_mat(self):
        i = 1
        while i <= (self._num_agents-1):
            j = i+1
            while j <= self._num_agents:
                index = (2*self._num_agents-i)*(i-1)/2+j-i
                self._A[i-1][j-1] = self._genes[round(index-1)]
                j += 1
            i += 1
        return self._A

    def get_total_cost(self):
        self._cost_tot = 0
        ad_mat = self.get_adj_mat()
        i = 1
        while i <= (self._num_agents-1):
            j = i+1
            while j <= self._num_agents:
                self._cost_tot += self._cost_par[i-1][j-1]*ad_mat[i-1][j-1]
                j += 1
            i += 1
        return self._cost_tot

    def get_fixed_cost(self):
        self._fixed_cost = sum(self._golden_genes)
        return self._fixed_cost

    def get_genes(self):
        return self._genes

    def get_golden_genes(self):
        return self._golden_genes

    def set_cost_mat(self, cost_mat):
        self._cost_par = cost_mat

    def get_fitness(self):
        self._fitness = 0
        functional = self.get_functional_nodes()
        deg_by_node = dict(self.get_graph().degree())
        deg_list = []
        for x in functional:
            deg_list.append(1 if deg_by_node.get(x, 0) == 3 else 0)
        summa = sum(deg_list)
        if self.test_degree_coord() and len(deg_list) > 0:
            self._fitness = summa / len(deg_list)
        return self._fitness

    def get_graph(self):
        self._graph.clear()
        self._adj_list = []
        for node in range(1, self._num_agents + 1):
            self._graph.add_node(node)
        i = 1
        while i <= (self._num_agents-1):
            row = []
            j = i+1
            while j <= self._num_agents:
                index = (2*self._num_agents-i)*(i-1)/2+j-i
                index = index-1
                gene_value = self._genes[round(index)]
                row.append(gene_value)
                if gene_value == 1:
                    self._graph.add_edge(i, j)
                j += 1
            self._adj_list.append(row)
            i += 1
        return self._graph

    def get_adj_list(self):
        return self._adj_list

    def set_constraint_org_team(self, node_list):
        if len(node_list) < 2:
            return
        self._master_nodes.append(node_list[0])
        self._int_constraints.append(node_list)
        master = node_list[0]
        for i in range(1, len(node_list)):
            a, b = min(master, node_list[i]), max(master, node_list[i])
            index = (2 * self._num_agents - a) * (a - 1) / 2 + (b - a)
            idx = round(index - 1)
            self._genes[idx] = 1
            self._golden_genes[idx] = 1
            self._team_edges.append([master, node_list[i]])

    def get_team_edges(self):
        return self._team_edges

    def get_constraints(self):
        return self._int_constraints

    def get_master_nodes(self):
        return self._master_nodes

    def set_constraint_org_hierarchy(self, node_list):
        self._master_nodes.append(node_list[0])
        self._int_constraints.append(node_list)
        i = 1
        while i < len(node_list):
            index = (2*self._num_agents-node_list[0])*(node_list[0]-1)/2+node_list[i]-node_list[0]
            self._genes[round(index-1)] = 1
            self._golden_genes[round(index-1)] = 1
            self._hierarchy_edges.append([node_list[0], node_list[i]])
            j = i + 1
            while j < len(node_list):
                index2 = (2*self._num_agents-node_list[i])*(node_list[i]-1)/2+node_list[j]-node_list[i]
                self._genes[round(index2-1)] = 0
                self._golden_genes[round(index2-1)] = 1
                j += 1
            i += 1

    def get_hierarchy_edges(self):
        return self._hierarchy_edges

    def get_functional_nodes(self):
        self._functional_nodes = []
        j = 2
        while j <= self._num_agents:
            if not self.is_coordinator(j):
                self._functional_nodes.append(j)
            j += 1
        return self._functional_nodes

    def test_degree_coord(self):
        self._degree_coord = False
        coordinators = self.get_master_nodes()
        deg = list(self.get_graph().degree().values())
        deg_coord = []
        for x in coordinators:
            if deg[x-1] >= 3:
                if deg[x-1] <= (((self._num_agents-1)/2)+1):
                    deg_coord.append(1)
                else:
                    deg_coord.append(0)
            else:
                deg_coord.append(0)
        summ = sum(deg_coord)
        if summ == len(deg_coord) and len(deg_coord) > 0:
            self._degree_coord = True
        return self._degree_coord

    def test_degree_funct(self):
        self._degree_funct = False
        functional = self.get_functional_nodes()
        deg = list(self.get_graph().degree().values())
        deg_funct = []
        for x in functional:
            if deg[x-1] >= 3:
                if deg[x-1] <= 3+int(self._num_agents/10):
                    deg_funct.append(1)
                else:
                    deg_funct.append(0)
            else:
                deg_funct.append(0)
        summ = 0
        for t in range(len(deg_funct)):
            if deg_funct[t] == 1:
                summ += 1
        if summ == len(deg_funct) and len(deg_funct) > 0:
            self._degree_funct = True
        return self._degree_funct

    def __str__(self):
        return self._genes.__str__()