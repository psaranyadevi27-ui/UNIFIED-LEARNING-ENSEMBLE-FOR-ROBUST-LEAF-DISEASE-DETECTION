# Fox Inspired Optimization Algorithm + Binary Waterwheel Plant optimization
# Hybrid FoxWheel Optimization Algorithm
import numpy as np

def hfwoa(Objective_function, lb, ub, pop_size, prob_size, epochs):
    population = np.random.uniform(lb, ub, size=(pop_size, prob_size))
    best_solution = None
    best_fitness = float('inf')
    lb = np.array(lb)
    ub = np.array(ub)
    for i in range(epochs):
        for j in range(pop_size):
            # bounds function
            population[j, population[j] < lb] = lb[population[j] < lb]
            population[j, population[j] > ub] = ub[population[j] > ub]
            fitness = Objective_function(population[j])

            if fitness < best_fitness:
                best_solution = population[j]
                best_fitness = fitness

            time_s = np.random.uniform(0, 1, prob_size)
            sp_s = best_solution / time_s
            dist_s = sp_s * time_s
            prey_dist = dist_s * 0.5
            r = np.random.uniform(0, 1)
            # Calculate tt (time transition)
            tt = sum(time_s) / prob_size
            minT = min(tt)
            # Calculate average time t
            t = tt / 2
            jump_s = 0.5 * 9.81 * (t)**2
            p = np.random.uniform(0, 1)
            c1 = np.random.uniform(0, 0.18)
            c2 = np.random.uniform(0.19, 1)
            r1 = np.random.uniform(0,2)
            r2 = np.random.uniform(0, 1)
            a = 2 * (i - 1/epochs)
            w = r1* (population[j] + 2*p)  # hybridization
            hyp = w* (2*p + r2)
            if r >= 0.5:
                # Exploitation
                if p > 0.18:
                    population[j] = prey_dist * jump_s * c1 + hyp
                elif p <=0.18:
                    population[j] = prey_dist * jump_s * c2 + hyp

            else:
                # Exploration
                population[j] = best_solution * np.random.rand(1, prob_size) * minT * a

    return best_solution, best_fitness