from __future__ import absolute_import

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    SIGMA2_NAME,
    LOWER_TRIANG_NAME,
    LENGTH_SCALE_NAME,
    PRODUCT_KERNELS_SEPARABLE,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel


def find_define_kernel_from_array(kernel_name):
    """

    :param kernel_name: (str) Name of the kernel
    :return: define_kernel_from_array associated to kernel_name
    """

    if kernel_name == MATERN52_NAME:
        return Matern52.define_kernel_from_array

    if kernel_name == TASKS_KERNEL_NAME:
        return TasksKernel.define_kernel_from_array

    raise NameError(kernel_name + " doesn't exist")

def find_kernel_constructor(kernel_name):
    """

    :param kernel_name: (str) Name of the kernel
    :return: kernel constructor
    """

    if kernel_name == MATERN52_NAME:
        return Matern52

    if kernel_name == TASKS_KERNEL_NAME:
        return TasksKernel

    raise NameError(kernel_name + " doesn't exist")

def define_prior_parameters_using_data(data, type_kernel, dimensions, sigma2_mean_matern52=None):
    """
    Defines value of the parameters of the prior distributions of the kernel's parameters.

    :param data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
    :param type_kernel: [str]
    :param dimensions: [int], It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product, and
            the total dimension of the product_kernels_separable too in the first entry.
    :param: sigma2_mean_matern52: float
    :return: {
        SIGMA2_NAME: float,
        LENGTH_SCALE_NAME: [float],
        LOWER_TRIANG_NAME: [float],
    }
    """

    # We assume that there is at most one task kernel, and mattern52 kernel in the product.

    parameters_priors = {
        SIGMA2_NAME: None,
        LENGTH_SCALE_NAME: None,
        LOWER_TRIANG_NAME: None,
    }

    index = -1

    if TASKS_KERNEL_NAME in type_kernel:
        index = type_kernel.index(TASKS_KERNEL_NAME)
        index_tasks = 0
        for i in xrange(1, index):
            index_tasks += dimensions[i]
        n_tasks = dimensions[index]
        tasks_index = data['points'][:, index_tasks]
        task_data = data.copy()
        task_data['points'] = tasks_index.reshape((len(tasks_index), 1))
        task_parameters = TasksKernel.define_prior_parameters(task_data, n_tasks)
        parameters_priors[LOWER_TRIANG_NAME] = task_parameters[LOWER_TRIANG_NAME]

    if MATERN52_NAME in type_kernel:
        m = data['points'].shape[1]
        n = data['points'].shape[0]
        indexes = [i for i in range(m) if i != index - 1]
        points_matern = data['points'][:, indexes]
        matern_data = data.copy()
        matern_data['points'] = points_matern
        matern52_parameters = Matern52.define_prior_parameters(matern_data, len(indexes),
                                                               var_evaluations=sigma2_mean_matern52)
        parameters_priors[SIGMA2_NAME] = matern52_parameters[SIGMA2_NAME]
        parameters_priors[LENGTH_SCALE_NAME] = matern52_parameters[LENGTH_SCALE_NAME]

    return parameters_priors

