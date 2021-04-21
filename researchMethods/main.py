#!/usr/bin/env python3
import sys
import numpy as np
from genetic import Genetic


def main():

    num_iteractions = []
    maxpop = 30
    numint = 20

    arg_selection = sys.argv[1]
    arg_cross_over = sys.argv[2]
    arg_mutation = sys.argv[3]
    arg_calculate_fitness = sys.argv[4]
    # main.py elitism two_points gaussian sphere
    # sys.argv =["main.py", "elitism", "two_points", "gaussian", "sphere"]
    filename = f"{arg_selection}_{arg_cross_over}_{arg_mutation}_{arg_calculate_fitness}.dat"

    if arg_calculate_fitness == 'sphere':
        criterion = -1
    if arg_calculate_fitness == 'rosenbrock':
        criterion = -100
    if arg_calculate_fitness == 'rastrigin':
        criterion = -100

    for k in range(5, maxpop + 1):
        t = 10*k

        iteractions_i = []
        for i in range(numint):
            GN = Genetic(30, t, sigma=0.1, selection=arg_selection,
                         cross_over=arg_cross_over, mutation=arg_mutation,
                         calculate_fitness=arg_calculate_fitness,
                         num_parents=int(t/10))
            GN.create_population()
            j = 0
            while True:
                GN.calculate_pop_fitness()
                best_fitness = GN.calculate_best_fitness()
                if best_fitness > criterion or j >= 500:
                    # print(j)
                    iteractions_i.append(j)
                    print(
                        f"POP: {t}\tTRY: {i}\tFITNESS: {best_fitness}\tNSTEPS: {j}; {iteractions_i}", file=sys.stderr)
                    print(t,"   ", j)
                    
                    break
                # =============seleção===================
                parents = GN.selection(GN.population)

                # print(f"Selected parents: {parents}", file=sys.stderr)

                # =============cross over=================
                GN.cross_over(GN.population, parents)

                # =============matação===================
                GN.mutation(GN.population)
                j = j + 1
        print()
        num_iteractions.append(iteractions_i)
        print(num_iteractions, file=sys.stderr)
    np.savetxt(filename,
               np.transpose(num_iteractions))


if __name__ == '__main__':
    main()
