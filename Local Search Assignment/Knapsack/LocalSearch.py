import copy
import math
import random

from matplotlib.pyplot import hist
from Item import Item


class LocalSearch:
    items: map
    linelength: int
    weight_limit: float

    def __init__(self, filename) -> None:
        self.import_file(filename)
        

    def import_file(self, filename):
        file = open(filename)
        for first_line in file:
            self.weight_limit = float(first_line)
            break
        self.linelength = 0
        self.items = {}
        for line in file:
            self.linelength +=1
            if self.linelength < 2: continue
            line = line.strip()
            line = line.split(',')
            if (len(line)) < 3: raise Exception('Error when providing data')
            item = Item(name=line[0], weight=float(line[1]), value=int(line[2]))
            self.items[item.name] = item

    
    def get_random_sol(self):
        list1 = [(key, random.randint(0, 1)) for key in self.items.keys()]
        while True:
            sum = 0
            for item_name, occurence in list1:
                item = self.items[item_name]
                sum +=item.weight * occurence
            # checks if the proposed list satisfies the max weight constraint

            if (sum < self.weight_limit):
                break
            else:
                list1 = [(key, random.randint(0, 1)) for key in self.items.keys()]
                
        return list1 # [(item, occurence)]
        
        
    def calc_tot_val(self, in_bag_items):
        tot_val = 0
        for item_name, occurence in in_bag_items:
            item = self.items[item_name]
            tot_val += item.value * occurence

        return tot_val

    
    def calc_tot_weight(self, in_bag_items):
        tot_weight = 0
        for item_name, occurence in in_bag_items:
            item = self.items[item_name]
            tot_weight += item.weight * occurence

        return tot_weight

    
    def neihbourgenerator(self, list, visted):
        currentlist = list
        # a list containing all possible neighbours
        grandlist = []
        # checks if a neighbour is already visited or nor

        for j in range(0, len(currentlist)):
            # shallow copy of the list
            copylist = copy.copy(currentlist) # [(str, occurence)]

            # flip a single bit for each neighbour
            for i in range(j, len(currentlist)):
                # if bit equals 0 flip to 1
                if (copylist[i][1] == 0):
                    copylist[i] = (copylist[i][0], 1)
                    break
                # if bit equals 1 flip to 0
                else:
                    copylist[i] = (copylist[i][0], 0)
                    break
            grandlist.append(copylist)

        # evaluate each neighbour for utility
        for i in range(len(grandlist)):
            # change state to a neighbour with the highest utility if such neighbour exists
            if (grandlist[i] not in visted and self.calc_tot_weight(grandlist[i]) <= self.weight_limit and self.calc_tot_val(grandlist[i]) > self.calc_tot_val(currentlist)
                    ):
                currentlist = grandlist[i]
                visted.append(grandlist[i])
            else:
                visted.append(grandlist[i])
                continue
        
        if currentlist == list: return None

        return currentlist


    def hillClimbingSearch(self, mylist=None):
        if not mylist:
            mylist = self.get_random_sol()
        visited = []
        while (True):
            # no better neighbour found
            neighbour = self.neihbourgenerator(mylist, visited)
            if (neighbour == None):
                return [self.calc_tot_val(mylist), mylist]
            else:
                mylist = neighbour


    def simulatedAnnealingSearch(self, temp=100, n=70):
        rand_sol = self.get_random_sol()
        history = []
        best_sol = rand_sol.copy()
        curr_sol = rand_sol.copy()
        for i in range(n):
            prop_sol = self.__rearrange_items(curr_sol)
            curr_val = self.calc_tot_val(curr_sol)
            prop_val = self.calc_tot_val(prop_sol)
            p = max(0, min(1, math.exp((prop_val - curr_val) / temp)))
            if prop_val > curr_val:
                p = 1
            if random.random() <= p:
                curr_sol = prop_sol.copy()
            
            if self.calc_tot_val(curr_sol) > self.calc_tot_val(best_sol):
                best_sol = curr_sol.copy()
            temp *=0.95
            history.append(self.calc_tot_val(curr_sol))
            
        return [self.calc_tot_val(best_sol), best_sol] # best solution, the jump history of the simulated annealing


    def __rearrange_items(self, rand_sol):
        while True:
            stop = random.randint(1, len(rand_sol))
            start = random.randint(0, stop)
            for i in range(start, stop):
                item_name, occurrence = rand_sol[i]
                if occurrence:
                    rand_sol[i] = (item_name, 0)
                else:
                    rand_sol[i] = (item_name, 1)
            if self.calc_tot_weight(rand_sol) <= self.weight_limit:
                break
        
        return rand_sol


    def crossover(self, parent1, parent2):
        trait_len = len(parent1)
        child = []
        for i in range(trait_len):
            if i > (trait_len //2):
                trait = parent2(i)
            else:
                trait = parent1[i]
            child.append(trait)



    def GA_search(self, init_popn_size=450, generations=50):
        n = init_popn_size
        popn = []
        for i in range(n):
            soln = self.get_random_sol()
            soln_val = self.calc_tot_val(soln)
            popn.append([soln_val, soln])

        popn.sort(key=lambda x:x[0])
        best_sol = popn[-1][1]
        for j in range(generations):
            new_popn = []
            for i in range(n-1, -1, -2):
                parent1 = popn[i][1]
                parent2 = popn[i-1][1]
                best_sol = parent1 if (popn[i][0] > self.fitness_function(best_sol)) else best_sol
                m = len(parent1)
                half1_parent1 = parent1[:m//2]
                half1_parent2 = parent2[:m//2]
                half2_parent1 = parent1[m//2:]
                half2_parent2 = parent2[m//2:]
                child1 = half1_parent1 + half2_parent2
                child2 = half1_parent2 + half2_parent1
                child1_fitness = self.fitness_function(child1)
                child2_fitness = self.fitness_function(child2)
                best_child = child1 if (self.fitness_function(child1) > self.fitness_function(child2)) else child2
                # print(self.fitness_function(best_sol))
                best_sol = best_child if (self.fitness_function(best_child) > self.fitness_function(best_sol)) else best_sol
                new_popn.append([child1_fitness, child1])
                new_popn.append([child2_fitness, child2])


            mutation_chance = random.random()
            if mutation_chance > 0.85:
                mutate_at_index = random.randint(0, len(popn))
                to_be_mutated_soln = new_popn[mutate_at_index][1]
                m = len(to_be_mutated_soln)
                index = mutate_at_index % m
                if to_be_mutated_soln[index][1]:
                    to_be_mutated_soln[index] = (to_be_mutated_soln[index][0], 0)
                    popn[mutate_at_index] = [self.fitness_function( to_be_mutated_soln),to_be_mutated_soln]
                else:
                    to_be_mutated_soln[index] = (to_be_mutated_soln[index][0], 1)
                    popn[mutate_at_index] = [self.fitness_function( to_be_mutated_soln),to_be_mutated_soln]

            new_popn.extend(popn)
            new_popn.sort(reverse=True, key=lambda x:x[0])
            # print(self.fitness_function(best_sol))
            popn = new_popn[:init_popn_size]

        print(self.fitness_function(best_sol))
        return [popn[0][0], popn[0][1]]


    def fitness_function(self, soln):
        val = self.calc_tot_val(soln)
        weight = self.calc_tot_weight(soln)
        return val if (weight < self.weight_limit) else 0