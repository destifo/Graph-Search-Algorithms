from collections import Counter
import copy
import math
from math import dist, radians, sin, cos, asin, sqrt
import random
import numpy as np
from Graph import Graph
import matplotlib.pyplot as plt

from matplotlib.pyplot import hist


class LocalSearch:
    cities: map
    graph: Graph
    geo_data: map
    linelength: int

    def __init__(self, nodefile, edgefile) -> None:
        self.graph = Graph()
        self.__build_graph(nodefile, edgefile)


    def __build_graph(self, nodefile, edgefile):
        self.geo_data = {} # {str:tuple}
        fh = open(nodefile)
        self.cities = []
        for line in fh:
            line = line.rstrip()
            line = line.split(',')
            self.cities.append(line[0])
            self.graph.addNode(line[0])
        self.geo_data[line[0]] = (line[1], line[2])

        fh = open(edgefile)
        for line in fh:
            line = line.split(',')
            self.graph.addEdge(line[0], line[1], int(line[2]))

    
    def get_random_sol(self):
        random_soln = []
        index = random.randint(0, len(self.cities)-1)
        start_city = self.cities[index]
        curr_city = start_city
        parent_of = {}
        random_soln.append(curr_city)
        visited = set()

        while len(visited) < len(self.cities):
            city = self.graph.nodes[curr_city]
            city_neighbours = list(city.neighbours.keys())
            picked_neighbour = None

            for i in range(len(city_neighbours)):
                neighbour = city_neighbours[i]
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                parent_of[neighbour] = curr_city
                picked_neighbour = neighbour
                break

            if not picked_neighbour:
                picked_neighbour = parent_of[curr_city]

            curr_city = picked_neighbour
            
            random_soln.append(picked_neighbour)


        return random_soln

    
    def calc_tot_distance(self, soln:list):
        tot_distance = 0
        n = len(soln)
        # print(soln)
        for i in range(n-1):
            city1 = soln[i]
            city2 = soln[i+1]
            # print(city1, city2)
            distance = 0
            # if city1 != city2:
            #     distance = self.graph.djikstraSearch(city1, city2)[0]
            # if distance == 0:   print(soln)
            tot_distance +=distance

        return tot_distance


    def hillClimbingSearch(self, start_soln=0):
        if not start_soln:
            start_soln = self.get_random_sol()

        curr_soln = start_soln
        best_sol = start_soln
        for i in range(50):
            while True:
                best_neighbour = self.find_best_neighbour(curr_soln)
                if not best_neighbour or self.fitness_function(best_sol) < self.fitness_function(best_neighbour):
                    break
                curr_soln = best_neighbour
                print("best soln updated with better val of:", self.fitness_function(best_sol))
                best_sol = curr_soln.copy()
            curr_soln = self.get_random_sol()

        return best_sol


    def find_best_neighbour(self, curr_soln:list, reverse_len=3):
        n = len(curr_soln)
        best_val = self.fitness_function(curr_soln)
        best_neighbour = None
        for j in range(2, n//2):
            for i in range(n-j):
                neighbour = curr_soln.copy()
                temp = neighbour[i]
                neighbour[i] = neighbour[i+j]
                neighbour[i+j] = temp
                neighbour_cost = self.fitness_function(neighbour)
                if neighbour_cost < best_val:
                    best_neighbour = neighbour
                    best_val = neighbour_cost

        return best_neighbour
        
        

    def simulatedAnnealingSearch(self, temp=100, n=70):
        rand_sol = self.get_random_sol()
        history = []
        best_sol = rand_sol.copy()
        curr_sol = rand_sol.copy()
        for i in range(n):
            prop_sol = self.get_random_sol()
            curr_distance = self.fitness_function(curr_sol)
            prop_distance = self.fitness_function(prop_sol)
            p = max(0, min(1, np.exp(-(prop_distance - curr_distance) / temp)))
            if prop_distance < curr_distance:
                p = 1
            if random.random() < p:
                curr_sol = prop_sol.copy()
            
            if self.fitness_function(curr_sol) < self.fitness_function(best_sol):
                best_sol = curr_sol.copy()
            temp *=0.9
            history.append(self.fitness_function(curr_sol))
            
        return best_sol, history # best solution, the jump history of the simulated annealing


    def __rearrange_cities(self, rand_sol):
        new_soln = rand_sol[:]
        stop = random.randint(1, len(new_soln))
        start = random.randint(0, stop)
        new_soln[start:stop] = reversed(new_soln[start:stop])
                  
        return new_soln


    def GA_search(self, init_popn_size=7, generations=90):
        n = init_popn_size
        popn = []
        best_sol = None
        best_val = float('inf')
        for i in range(n):
            soln = self.get_random_sol()
            soln_val = self.fitness_function(soln)
            if soln_val < best_val:
                best_val = soln_val
                best_sol = soln
            popn.append([soln_val, soln])
        
        # for entity in popn:
        #     print(entity)

        for j in range(generations):
            new_popn = []
            for i in range(n-1, -1, -2):
                # print(self.calc_tot_distance(best_sol), best_sol)
                parent1 = popn[i][1]
                parent2 = popn[i-1][1]
                # best_sol = parent1 if (popn[i][0] < self.fitness_function(best_sol)) else best_sol
                m = len(parent1)
                half1_parent1 = parent1[:m//2]
                half1_parent2 = parent2[:m//2]
                half2_parent1 = parent1[m//2:]
                half2_parent2 = parent2[m//2:]
                child1 = half1_parent1 + half2_parent2
                child2 = half1_parent2 + half2_parent1
                if not self.has_all_city(child1) and not self.has_all_city(child2):
                    continue
                elif not self.has_all_city(child1):
                    child1 = parent1
                elif not self.has_all_city(child2):
                    child2 = parent1
                child1_fitness = self.fitness_function(child1)
                child2_fitness = self.fitness_function(child2)
                best_child = child1 if (self.fitness_function(child1) < self.fitness_function(child2)) else child2
                # print(self.fitness_function(best_sol))
                best_sol = best_child if (self.fitness_function(best_child) < self.fitness_function(best_sol)) else best_sol
                new_popn.append([child1_fitness, child1])
                new_popn.append([child2_fitness, child2])


            # print(self.calc_tot_distance(best_sol), best_sol)
            new_popn.extend(popn)
            mutation_chance = random.random()
            if mutation_chance > 0.85:
                mutate_at_index = random.randint(0, len(new_popn)-1)
                to_be_mutated_soln = new_popn[mutate_at_index][1]
                m = len(to_be_mutated_soln)
                index = mutate_at_index % m
                mutated_soln = self.swap_random_cities(to_be_mutated_soln)
                while not self.has_all_city(mutated_soln):
                    mutated_soln = self.swap_random_cities(to_be_mutated_soln)
                new_popn[mutate_at_index] = ([self.fitness_function(mutated_soln), mutated_soln])
            new_popn.sort(key=lambda x:x[0])
            # print(self.fitness_function(best_sol))
            popn = new_popn[:init_popn_size]

        return popn[0][0], popn[0][1]   


    def swap_random_cities(self, soln):
        n = len(soln)
        rand_index1 = random.randint(0, n-1)
        rand_index2 = random.randint(0, n-1)
        while rand_index1 == rand_index2:
            rand_index1 = random.randint(0, n-1)
            rand_index2 = random.randint(0, n-1)

        temp = soln[rand_index1]
        soln[rand_index1] = soln[rand_index2]
        soln[rand_index2] = temp

        return soln


    def find_repeated_cities(self, soln):
        city_count =Counter(soln)
        repn_count = 0
        for city, repn in city_count.items():
            repn_count += (repn - 1)
        
        return repn_count


    def has_all_city(self, soln):
        return self.calc_city_num(soln) == len(self.cities)


    def calc_city_num(self, soln):
        cities_in_soln = set()
        for city in soln:
            cities_in_soln.add(city)

        return len(cities_in_soln)

    
    def fitness_function(self, soln:list):
        distance = self.calc_tot_distance(soln)
        repn = self.find_repeated_cities(soln)

        n = len(self.cities) / 10
        fitness_val = distance * (repn/n)

        return fitness_val


    def crossover(self, parent1, parent2):
        trait_len = len(parent1)
        child = []
        for i in range(trait_len):
            if i > (trait_len //2):
                trait = parent2(i)
            else:
                trait = parent1[i]
            child.append(trait)


    # generate the realistic path of a solution
    def generate_full_path(self, path:list):
        full_path = []
        n = len(path)
        for i in range(n-1):
            city1 = path[i]
            city2 = path[i+1]

            if city1 == city2:
                full_path.append(city1)
            else:
                between_path = self.graph.djikstraSearch(city1, city2)[1]
                cities = between_path.split('->')
                print(cities)
                m = len(cities)
                for j in range(m-1):
                    full_path.append(cities[j])
        
        full_path.append(path[-1])

        return full_path


    def plot(self, soln, i=''):
        lat = []
        lon = []
        for city in soln:
            lat.append(self.geo_data[city][0])
            lon.append(self.geo_data[city][1])
        
        plt.scatter(lat, lon, marker='x')