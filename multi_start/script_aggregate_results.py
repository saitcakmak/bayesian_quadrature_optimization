from __future__ import absolute_import

import numpy as np
import os


import matplotlib.pyplot as plt
plt.switch_backend('agg')

import argparse
from stratified_bayesian_optimization.util.json_file import JSONFile

def get_best_values(data, n_restarts=9, n_training=3, sign=False):
    chosen = data['chosen_index']
    current_value = []
    # TODO: TEMPORAL CHANGE
    for i in range(n_restarts):
        current_value.append(data['evaluations'][str(i)][n_training - 1][0])

    index_points = {}

    for i in range(n_restarts):
        index_points[i] = n_training
    best_values = []
    best_values.append(np.max(current_value))
    n_iterations = len(chosen)

    for i in range(n_iterations):
        j = int(data['chosen_index'][i])
        if sign:

            current_value[j] = -1.0 * data['evaluations'][str(j)][index_points[j]][0]
        else:
            current_value[j] = data['evaluations'][str(j)][index_points[j]][0]

        best_values.append(np.max(current_value))
        index_points[j] += 1
    return best_values

if __name__ == '__main__':
    # python -m multi_start.script_aggregate_results approx_lipschitz problem5 20 50 1 20

    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='approx_lipschitz')
    parser.add_argument('problem_name', help='analytic_example')
    parser.add_argument('n_starting_points', help=20)
    parser.add_argument('n_iterations', help=100)
    parser.add_argument('lower_random_seed', help=1)
    parser.add_argument('upper_random_seed', help=100)
   # parser.add_argument('method_2', help='real_gradient')


    args_ = parser.parse_args()
    n_iterations = int(args_.n_iterations)
    method = args_.method
    problem = args_.problem_name
    n_restarts = int(args_.n_starting_points)
    lower_random_seed = int(args_.lower_random_seed)
    upper_random_seed = int(args_.upper_random_seed)


    prefix_file_1 = 'data/multi_start/' + problem + '/' + 'greedy_policy/' + method + '_random_seed_'
    prefix_file_2 = 'data/multi_start/' + problem + '/' +'uniform_policy/' + method + '_random_seed_'
   # prefix_file_3 = 'data/multi_start/' + problem + '/' + 'random_policy/' + method + '_random_seed_'
    prefix_file_3 = 'data/multi_start/' + problem + '/' + 'swersky_greedy_policy/' + 'swersky' + '_random_seed_'

    data = {}
    data_2 = {}
    data_3 = {}
    for i in range(lower_random_seed, upper_random_seed):
        file_1 = prefix_file_1 + str(i) + '_n_restarts_' + str(n_restarts) + '.json'
        try:
            data[i] = JSONFile.read(file_1)
        except Exception as e:
            data[i] = None

        file_2 = prefix_file_2 + str(i) + '_n_restarts_' + str(n_restarts)+ '.json'
        try:
            data_2[i] = JSONFile.read(file_2)
        except Exception as e:
            data_2[i] = None

        file_3 = prefix_file_3 + str(i) + '_n_restarts_' + str(n_restarts)+ '.json'

        try:
            data_3[i] = JSONFile.read(file_3)
        except Exception as e:
            data_3[i] = None


    type_1 = 'MLS' #+ method
   # types = [type_1, 'equal_allocation', 'random_allocation']
    types = [type_1, 'equal_allocation', 'swk']

  #  types = [types[0]]

    # temporal change from 3 to 4
    n_training = 3
    n = n_iterations

    best_values_rs = {}

    for t in types:
        best_values_rs[t] = {}

    for i in range(lower_random_seed, upper_random_seed):
        data_list = [data[i], data_2[i], data_3[i]]
        best_values = {}
        data_dict = {}

        data_dict[type_1] = data[i]
        data_dict['equal_allocation'] = data_2[i]
        data_dict['swk'] = data_3[i]

        for t in types:
            if data_dict[type_1] is not None and data_dict['equal_allocation'] is not None and data_dict['swk'] is not None:
                best_values[t] = get_best_values(data_dict[t], n_restarts, n_training)

                for r in range(len(best_values[t])):
                    if r not in best_values_rs[t]:
                        best_values_rs[t][r] = []
                    best_values_rs[t][r].append(best_values[t][r])

    best_values = {}
    iterations = n

    ci_lower = {}
    ci_upper = {}
    n_elements = {}
    for t in types:
        best_values[t] = []
        ci_lower[t] = []
        ci_upper[t] = []
        n_elements[t] = []
        for i in range(iterations):
            value = np.mean(best_values_rs[t][i])
            best_values[t].append(value)
            std = np.std(best_values_rs[t][i])
            std /= np.sqrt(float(len(best_values_rs[t][i])))
            ci_lower[t].append(value - 1.96 * std)
            ci_upper[t].append(value + 1.96 * std)
            n_elements[t].append(len(best_values_rs[t][i]))

    for t in types:
        print t
        print (n_elements[t])
    points = range(n_training, n_training + n)

    plt.figure()
    colors = {}
    colors[types[0]] = 'b'

    if len(types) == 2:
        colors[types[1]] = 'r'

    if len(types) == 3:
        colors[types[1]] = 'r'
        colors[types[2]] = 'g'

    for t in types:
        z = best_values[t][0:n]
        lci = ci_lower[t][0:n]
        uci = ci_upper[t][0:n]
        plt.plot(points[0:n], z,  colors[t], label=t)
        plt.plot(points[0:n], lci, colors[t] + '--')
        plt.plot(points[0:n], uci, colors[t] + '--')

    plt.legend()
    plt.ylabel('Best Value')
    plt.xlabel('Iteration')

    file_name = 'data/multi_start/aggregated_plot_policies/'

    if not os.path.exists(file_name):
        os.mkdir(file_name)

    file_name += problem + '/'

    if not os.path.exists(file_name):
        os.mkdir(file_name)


    file_name += type_1 + '_random_'  + '_uniform' + '_' + 'swk_'+ 'best_solution'

    plt.savefig(file_name + '.pdf')