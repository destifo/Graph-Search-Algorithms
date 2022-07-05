import copy
import math
from math import radians, sin, cos, asin, sqrt
import random
import numpy as np

from matplotlib.pyplot import hist


class LocalSearch:
    cities: map
    geo_data: map
    linelength: int

    def __init__(self, filename) -> None:
        self.import_file(filename)
        

    def import_file(self, filename):
        self.geo_data = {} # {str:tuple}
        fh = open(filename)
        self.cities = []
        for line in fh:
            line = line.rstrip()
            line = line.split(',')
            self.cities.append(line[0])

            self.geo_data[line[0]] = (line[1], line[2])

    
    def fitness_function(self, soln:list):
        tot_distance = 0
        n = len(self.cities)

        for i in range(0, n-1):
            start_city = soln[i]
            final_city = soln[i+1]
            tot_distance +=self.calc_distance(start_city, final_city)
            if i == n-2:
                start_city = soln[-1]
                final_city = soln[0]
                tot_distance +=self.calc_distance(start_city, final_city)

        return round(tot_distance, 2)

    
    def get_random_sol(self):
        random_soln = random.sample(self.cities, len(self.cities))

        return random_soln
        
        
    def calc_distance(self, initial, final):
        lon1 = radians(eval(self.geo_data[initial][1]))
        lon2 = radians(eval(self.geo_data[final][1]))
        lat1 = radians(eval(self.geo_data[initial][0]))
        lat2 = radians(eval(self.geo_data[final][0]))
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
        c = 2 * asin(sqrt(a))
        r = 6371

        return(c * r)

    
    # def find_neighbours(self, start, visited_set):
    #     neighbours = []
    #     for city in self.cities:
    #         if city == start or city in visited_set:
    #             continue
    #         travel_cost = self.calc_distance(start, city)
    #         neighbours.append((travel_cost, city))
        
    #     return neighbours


    def hillClimbingSearch(self, start_soln=0):
        if not start_soln:
            start_soln = self.get_random_sol()
        
        best_sol = start_soln
        while True:
            best_neighbour = self.find_neighbours(best_sol)
            if not best_neighbour:
                path = best_sol
                distance = self.fitness_function(path)
                return distance, path
            best_sol = best_neighbour


    def find_neighbours(self, curr_soln:list, reverse_len=3):
        neighbours = []
        n = len(curr_soln)
        best_neighbour = curr_soln
        for i in range(1, n-reverse_len):
            new_soln = curr_soln.copy()
            new_soln[i:i+reverse_len] = reversed(new_soln[i:i+reverse_len])
            if self.fitness_function(new_soln) < self.fitness_function(best_neighbour): best_neighbour = new_soln
            neighbours.append(new_soln)

        return best_neighbour if (best_neighbour != curr_soln) else None
        

    def simulatedAnnealingSearch(self, temp=100, n=200):
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
            temp /=(i+1)*0.05
            history.append(self.fitness_function(curr_sol))
            
        return best_sol, history # best solution, the jump history of the simulated annealing


    def __rearrange_cities(self, rand_sol):
        new_soln = rand_sol[:]
        stop = random.randint(1, len(new_soln))
        start = random.randint(0, stop)
        new_soln[start:stop] = reversed(new_soln[start:stop])
                  
        return new_soln


    def GA_search(self, init_popn_size=400, generations=70):
        n = init_popn_size
        popn = []
        for i in range(n):
            soln = self.get_random_sol()
            soln_val = self.fitness_function(soln)
            popn.append([soln_val, soln])

        popn.sort(key=lambda x:x[0])
        best_sol = popn[-1][1]
        for j in range(generations):
            new_popn = []
            for i in range(n-1, -1, -2):
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
                child1_fitness = self.fitness_function(child1)
                child2_fitness = self.fitness_function(child2)
                best_child = child1 if (self.fitness_function(child1) < self.fitness_function(child2)) else child2
                # print(self.fitness_function(best_sol))
                best_sol = best_child if (self.fitness_function(best_child) < self.fitness_function(best_sol)) else best_sol
                new_popn.append([child1_fitness, child1])
                new_popn.append([child2_fitness, child2])


            # mutation_chance = random.random()
                # if mutation_chance > 0.85:
                #     mutate_at_index = random.randint(0, len(popn))
                #     to_be_mutated_soln = new_popn[mutate_at_index][1]
                #     m = len(to_be_mutated_soln)
                #     index = mutate_at_index % m
                #     if to_be_mutated_soln[index][1]:
                #         to_be_mutated_soln[index] = (to_be_mutated_soln[index][0], 0)
                #         popn[mutate_at_index] = [self.calc_tot_val(to_be_mutated_soln[index], to_be_mutated_soln[index])]
                #     else:
                #         to_be_mutated_soln[index] = (to_be_mutated_soln[index][0], 1)
                #         popn[mutate_at_index] = [self.calc_tot_val(to_be_mutated_soln[index], to_be_mutated_soln[index])]

            new_popn.extend(popn)
            new_popn.sort(key=lambda x:x[0])
            # print(self.fitness_function(best_sol))
            popn = new_popn[:init_popn_size]

        return popn[0][0], popn[0][1]    


    def crossover(self, parent1, parent2):
        trait_len = len(parent1)
        child = []
        for i in range(trait_len):
            if i > (trait_len //2):
                trait = parent2(i)
            else:
                trait = parent1[i]
            child.append(trait)