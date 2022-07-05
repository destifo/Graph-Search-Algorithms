import sys
from LocalSearch2 import LocalSearch


inputs =sys.argv
def readGraphFromCommand():
    if len(inputs)!=5:
        raise Exception("Invalid input")
    
    edgefilename = inputs[4]
    ls = LocalSearch(edgefile=edgefilename)
    if inputs[2] =="ga":
        ga_soln_fitness, GA_soln = ls.GA_search()
        ga_soln_path_cost = ls.calc_tot_distance(GA_soln)
        print("final GA solution fitness value:", ga_soln_fitness)
        print("final GA solution path cost:", ga_soln_path_cost)
        print(GA_soln)
        return GA_soln
    elif inputs[2] =="hc":
        hc_sol = ls.hillClimbingSearch()
        print("HC solution:", hc_sol)
        hc_sol_cost = ls.fitness_function(hc_sol)
        print("final solution fitness value:", hc_sol_cost)
        print("final solution path cost:", ls.calc_tot_distance(hc_sol))
        return hc_sol
    elif inputs[2] =="sa":
        simulated_annealing_soln, history = ls.simulatedAnnealingSearch()
        fitness_val = ls.fitness_function(simulated_annealing_soln)
        soln_path_cost = ls.calc_tot_distance(simulated_annealing_soln)
        print("SA solution:", simulated_annealing_soln)
        print()
        print("history of the accepted propsed solution values:", history)
        print()
        print("final solution fitness value:", fitness_val)
        print("final solution path cost:", soln_path_cost)
        return simulated_annealing_soln


readGraphFromCommand()