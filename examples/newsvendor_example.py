import sys

# Add the repo path
sys.path.append('/Users/saitcakmak/Documents/bayesian_quadrature_optimization')
from stratified_bayesian_optimization.services.bgo import bgo
from scipy.stats import gamma
from problems.vendor_problem.main import *


def f(x, n_samples):
    return main(n_samples, x)


def g(x):
    return main_objective(1, x)


customers = 1000
lower_bound = gamma.ppf(.001, customers)
upper_bound = gamma.ppf(.999, customers)

bounds_domain_x = [(0, customers), (0, customers)]
bounds_domain_w = [[lower_bound, upper_bound], [lower_bound, upper_bound]]
type_bounds = [0, 0, 0, 0]
name_method = 'bqo'
n_iterations = 2
random_seed = 2
n_training = 4
n_restarts = 2
n_restarts_mean = 1
n_best_restarts_mean = 0
maxepoch = 10
thinning = 5
n_burning = 30
maxepoch_mean = 10
default_n_samples_parameters = 5
default_n_samples = 5

distribution = 'gamma'
noise = True
n_samples_noise = 5
parameters_distribution = {'scale': [1.0], 'a': [customers]}

sol = bgo(
    g, bounds_domain_x, integrand_function=f, bounds_domain_w=bounds_domain_w, type_bounds=type_bounds,
    name_method=name_method, n_iterations=n_iterations, random_seed=random_seed, n_training=n_training,
    n_restarts=n_restarts, n_restarts_mean=n_restarts_mean, n_best_restarts_mean=n_best_restarts_mean,
    maxepoch=maxepoch, thinning=thinning, n_burning=n_burning, maxepoch_mean=maxepoch_mean,
    default_n_samples_parameters=default_n_samples_parameters, default_n_samples=default_n_samples,
    distribution=distribution, noise=noise, n_samples_noise=n_samples_noise,
    parameters_distribution=parameters_distribution)

print(sol)
