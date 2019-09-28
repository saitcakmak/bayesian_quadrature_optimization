import sys
# Add the repo path
sys.path.append('/Users/saitcakmak/Documents/bayesian_quadrature_optimization')
from stratified_bayesian_optimization.services.bgo import bgo
from problems.arxiv.main import *


def f(x):
    return main(x)


def g(x):
    return main_objective(x)


bounds_domain_x = [(0.01, 5.0), (0.0, 2.1), (1, 21), (1, 201)]
bounds_domain_w = [[0, 1, 2, 3, 4]]
type_bounds = [0, 0, 0, 0, 1]
name_method = 'bqo'
n_iterations = 2
random_seed = 1
n_training = 5
n_restarts = 2
n_restarts_mean = 1
n_best_restarts_mean = 0
maxepoch = 10
thinning = 5
n_burning = 30
maxepoch_mean = 10
default_n_samples_parameters = 5
default_n_samples = 5

sol = bgo(
    g, bounds_domain_x, integrand_function=f, bounds_domain_w=bounds_domain_w, type_bounds=type_bounds,
    name_method=name_method, n_iterations=n_iterations, random_seed=random_seed, n_training=n_training,
    n_restarts=n_restarts, n_restarts_mean=n_restarts_mean, n_best_restarts_mean=n_best_restarts_mean,
    maxepoch=maxepoch, thinning=thinning, n_burning=n_burning, maxepoch_mean=maxepoch_mean,
    default_n_samples_parameters=default_n_samples_parameters, default_n_samples=default_n_samples)

print(sol)
