import sys
from LocalSearch import LocalSearch


inputs =sys.argv
def readGraphFromCommand():
    if len(inputs)!=5:
        raise Exception("Invalid input")
    
    filename = inputs[4]
    ls = LocalSearch(filename)
    if inputs[2] =="sa":
        sa_algo_soln_val, sa_algo_soln = ls.simulatedAnnealingSearch()
        print("solution:", sa_algo_soln)
        print("Total Value:", sa_algo_soln_val)
        print("Total weight:", ls.calc_tot_weight(sa_algo_soln))
        return sa_algo_soln
    elif inputs[2] =="hc":
        sol_val, sol = ls.hillClimbingSearch()
        print("solution:", sol)
        print("Total Value:", sol_val)
        print("Total weight:", ls.calc_tot_weight(sol))
        return sol
    elif inputs[2] =="ga":
        ga_algo_soln_val, ga_algo_soln = ls.GA_search()
        print("solution:", ga_algo_soln)
        print("Total Value:", ga_algo_soln_val)
        print("Total weight:", ls.calc_tot_weight(ga_algo_soln))
        return ga_algo_soln

readGraphFromCommand()