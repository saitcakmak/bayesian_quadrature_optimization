from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.util import (
    get_number_parameters_kernel,
    convert_dictionary_gradient_to_simple_dictionary,
)
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    LENGTH_SCALE_NAME,
    LARGEST_NUMBER,
    SMALLEST_POSITIVE_NUMBER,
)
from stratified_bayesian_optimization.priors.uniform import UniformPrior


class Matern52(AbstractKernel):

    def __init__(self, dimension, length_scale, **kernel_parameters):
        """
        :param dimension: int
        :param length_scale: ParameterEntity
        """

        name = MATERN52_NAME
        dimension_parameters = get_number_parameters_kernel([name], [dimension])

        super(Matern52, self).__init__(name, dimension, dimension_parameters)

        self.length_scale = length_scale

    @property
    def hypers(self):
        return {
            self.length_scale.name: self.length_scale,
        }

    @property
    def hypers_as_list(self):
        """
        This function defines the default order of the parameters.
        :return: [ParameterEntity]
        """

        return [self.length_scale]

    @property
    def hypers_values_as_array(self):
        """

        :return: np.array(n)
        """
        parameters = []
        parameters.append(self.length_scale.value)

        return np.concatenate(parameters)

    def sample_parameters(self, number_samples, random_seed=None):
        """

        :param number_samples: (int) number of samples
        :param random_seed: int
        :return: np.array(number_samples x k)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        samples = []
        parameters = [self.length_scale]
        for parameter in parameters:
            samples.append(parameter.sample_from_prior(number_samples))
        return np.concatenate(samples, 1)

    def get_bounds_parameters(self):
        """
        Return bounds of the parameters of the kernel
        :return: [(float, float)]
        """
        bounds = []
        parameters = [self.length_scale]
        for parameter in parameters:
            bounds += parameter.bounds
        return bounds

    @property
    def name_parameters_as_list(self):
        """

        :return: ([(name_param, name_params)]) name_params can be other list if name_param
            represents several parameters (like an array), otherwise name_params=None.
        """
        return [(self.length_scale.name, [(i, None) for i in xrange(self.dimension)])]

    def set_parameters(self, length_scale=None):
        """

        :param length_scale: ParameterEntity
        """
        if length_scale is not None:
            self.length_scale = length_scale

    def update_value_parameters(self, params):
        """

        :param params: np.array(n)
        """
        self.length_scale.set_value(params[0:self.dimension])

    @classmethod
    def define_kernel_from_array(cls, dimension, params, **kernel_parameters):
        """
        :param dimension: (int) dimension of the domain of the kernel
        :param params: (np.array(k)) The first part are the parameters for length_scale.

        :return: Matern52
        """

        length_scale = ParameterEntity(LENGTH_SCALE_NAME, params[0:dimension], None)

        return cls(dimension, length_scale)

    @classmethod
    def define_default_kernel(cls, dimension, bounds=None, default_values=None,
                              parameters_priors=None, **kernel_parameters):
        """
        :param dimension: (int) dimension of the domain of the kernel
        :param bounds: [[float, float]], lower bound and upper bound for each entry of the domain.
            This parameter is used to compute priors in a smart way.
        :param default_values: (np.array(k)) The first part are the parameters for length_scale
        :param parameters_priors: {
            LENGTH_SCALE_NAME: [float],
        }

        :return: Matern52
        """

        if parameters_priors is None:
            parameters_priors = {}

        if default_values is None:
            ls = parameters_priors.get(LENGTH_SCALE_NAME, dimension * [1.0])
            default_values = ls

        kernel = cls.define_kernel_from_array(dimension, default_values)


        if bounds is not None:
            # a= 0.05 0.324
            diffs = [float(bound[1] - bound[0]) / 0.001 for bound in bounds]
            prior = UniformPrior(dimension, dimension * [SMALLEST_POSITIVE_NUMBER], diffs)
            bounds = [(SMALLEST_POSITIVE_NUMBER, diff) for diff in diffs]
        else:
            diffs = parameters_priors.get(LENGTH_SCALE_NAME, dimension * [LARGEST_NUMBER])
            prior = UniformPrior(dimension, dimension * [SMALLEST_POSITIVE_NUMBER], diffs)
            bounds = [(SMALLEST_POSITIVE_NUMBER, diff) for diff in diffs]

        kernel.length_scale.prior = prior
        kernel.length_scale.bounds = bounds

        return kernel

    def cov(self, inputs):
        """

        :param inputs: np.array(nxd)
        :return: np.array(nxn)
        """
        return self.cross_cov(inputs, inputs)

    def cross_cov(self, inputs_1, inputs_2):
        """

        :param inputs_1: np.array(nxd)
        :param inputs_2: np.array(mxd)
        :return: np.array(nxm)
        """
        r2 = np.abs(Distances.dist_square_length_scale(self.length_scale.value, inputs_1, inputs_2))
        r = np.sqrt(r2)
        cov = (1.0 + np.sqrt(5)*r + (5.0/3.0)*r2) * np.exp(-np.sqrt(5)*r)
        return cov

    def gradient_respect_parameters(self, inputs):
        """

        :param inputs: np.array(nxd)
        :return: {
            'length_scale': {'entry (int)': nxn},
        }
        """
        grad = GradientLSMatern52.gradient_respect_parameters_ls(inputs, self.length_scale)

        return grad

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: np.array(1xd)
        :param inputs: np.array(nxd)

        :return: np.array(nxd)
        """
        grad = GradientLSMatern52.grad_respect_point(self.length_scale, point, inputs)
        return grad

    def hessian_respect_point(self, point, inputs):
        """
        Computes the hessians of cov(point, inputs) respect point

        :param point:
        :param inputs:
        :return: np.array(nxdxd)
        """
        hessian = GradientLSMatern52.hessian_respect_point(self.length_scale, point, inputs)
        return hessian

    @classmethod
    def evaluate_grad_respect_point(cls, params, point, inputs, dimension):
        """
        Evaluate the gradient of the kernel defined by params respect to the point.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param point: np.array(1xd)
        :param inputs: np.array(nxd)
        :param dimension: (int) dimension of the domain of the kernel
        :return: np.array(nxd)

        """
        matern52 = cls.define_kernel_from_array(dimension, params)
        return matern52.grad_respect_point(point, inputs)

    @classmethod
    def evaluate_hessian_respect_point(cls, params, point, inputs, dimension):
        """
        Evaluate the hessian of the kernel defined by params respect to the point.

        :param params:
        :param point:
        :param inputs:
        :param dimension: int
        :return:
        """
        matern52 = cls.define_kernel_from_array(dimension, params)
        return matern52.hessian_respect_point(point, inputs)

    @classmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension, **kwargs):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :return: (np.array(nxn)) cov(inputs) where the kernel is defined with params
        """
        matern52 = cls.define_kernel_from_array(dimension, params)
        return matern52.cov(inputs)

    @classmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension, **kwargs):
        """
        Evaluate the gradient respect the parameters of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :return: {
            (int) i: (nxn), derivative respect to the ith parameter
        }
        """
        matern52 = cls.define_kernel_from_array(dimension, params)
        gradient = matern52.gradient_respect_parameters(inputs)

        names = matern52.name_parameters_as_list

        gradient = convert_dictionary_gradient_to_simple_dictionary(gradient, names)
        return gradient

    @classmethod
    def evaluate_cross_cov_defined_by_params(cls, params, inputs_1, inputs_2, dimension, **kwargs):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param inputs_1: np.array(nxm)
        :param inputs_2: np.array(kxm)
        :param dimension: (int) dimension of the domain of the kernel

        :return: (np.array(nxk)) cov(inputs_1, inputs_2) where the kernel is defined with params
        """
        matern52 = cls.define_kernel_from_array(dimension, params)
        return matern52.cross_cov(inputs_1, inputs_2)

    @staticmethod
    def define_prior_parameters(data, dimension):
        """
        Defines value of the parameters of the prior distributions of the kernel's parameters.

        :param data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}. Each point is the is an index of the task.
        :param dimension: int
        :return:  {
            LENGTH_SCALE_NAME: [float],
        }
        """
        # Take mean value of |x-y| for all points x,y in the training data. I think that it's a
        # good starting value for the parameter ls in the kernel.

        diffs_training_data_x = []
        n = data['points'].shape[0]

        for i in xrange(dimension):
            if n > 1:
                points = [abs(data['points'][j, i] - data['points'][h, i])
                          for j in xrange(n) for h in xrange(n)]
            else:
                points = [0.324]

            diffs_training_data_x.append(np.mean(points) / 0.324)

        return {
            LENGTH_SCALE_NAME: diffs_training_data_x,
        }

    @staticmethod
    def compare_kernels(kernel1, kernel2):
        """
        Compare the values of kernel1 and kernel2. Returns True if they're equal, otherwise it
        return False.

        :param kernel1: Matern52 instance object
        :param kernel2: Matern52 instance object
        :return: boolean
        """
        if kernel1.name != kernel2.name:
            return False

        if kernel1.dimension != kernel2.dimension:
            return False

        if kernel1.dimension_parameters != kernel2.dimension_parameters:
            return False

        if np.any(kernel1.length_scale.value != kernel2.length_scale.value):
            return False

        return True

    @staticmethod
    def parameters_from_list_to_dict(params, **kwargs):
        """
        Converts a list of parameters to dictionary using the order of the kernel.

        :param params: [float]

        :return: {
            LENGTH_SCALE_NAME: [float],
        }
        """

        parameters = {}

        parameters[LENGTH_SCALE_NAME] = params

        return parameters


class GradientLSMatern52(object):

    @classmethod
    def gradient_respect_parameters_ls(cls, inputs, ls):
        """

        :param inputs: np.array(nxd)
        :param ls: (ParameterEntity) length_scale
        :return: {
            'length_scale': {'entry (int)': nxn}
        }
        """

        derivate_respect_to_r = cls.gradient_respect_distance(ls, inputs)

        grad = {}
        grad[ls.name] = {}

        grad_distance_ls = Distances.gradient_distance_length_scale_respect_ls(ls.value, inputs)

        for i in range(ls.dimension):
            grad[ls.name][i] = grad_distance_ls[i] * derivate_respect_to_r

        return grad

    @classmethod
    def gradient_respect_distance(cls, ls, inputs):
        """
        :param ls: (ParameterEntity) length_scale
        :param inputs: np.array(nxd)
        :return: np.array(nxn)
        """

        return cls.gradient_respect_distance_cross(ls, inputs, inputs)

    @classmethod
    def gradient_respect_distance_cross(cls, ls, inputs_1, inputs_2, second=False):
        """

        :param ls: (ParameterEntity) length_scale
        :param inputs_1: np.array(nxd)
        :param inputs_2: np.array(mxd)
        :param second: (boolean) Computes second derivative if it's True.
        :return: np.array(nxm) or {'first': np.array(nxm), 'second': np.array(nxm)}
        """

        r2 = np.abs(Distances.dist_square_length_scale(ls.value, inputs_1, inputs_2))
        r = np.sqrt(r2)

        exp_r = np.exp(-np.sqrt(5) * r)

        part_1 = (1.0 + np.sqrt(5) * r + (5.0/3.0) * r2) * exp_r * (-np.sqrt(5))
        part_2 = (exp_r * (np.sqrt(5) + (10.0/3.0) * r))
        derivate_respect_to_r = part_1 + part_2

        if not second:
            return derivate_respect_to_r

        part_0 = (10.0 / 3.0) * exp_r
        part_1 = part_1 * (-np.sqrt(5.0))
        part_3 = 2.0 * ( (10.0 / 3.0) * r + np.sqrt(5.0)) * exp_r * (-np.sqrt(5.0))

        hessian_respect_to_r = part_0 + part_1 + part_3

        sol = {
            'first': derivate_respect_to_r,
            'second': hessian_respect_to_r,
        }

        return sol

    @classmethod
    def grad_respect_point(cls, ls, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param ls: (ParameterEntity) length_scale
        :param point: np.array(1xd)
        :param inputs: np.array(nxd)

        :return: np.array(nxd)
        """

        derivate_respect_to_r = cls.gradient_respect_distance_cross(ls, point, inputs)
        grad_distance_point = \
            Distances.gradient_distance_length_scale_respect_point(ls.value, point, inputs)

        gradient = grad_distance_point * derivate_respect_to_r.transpose()

        gradient = np.nan_to_num(gradient)

        return gradient


    @classmethod
    def hessian_respect_point(cls, ls, point, inputs):
        """
        Computes the Hessians of cov(point, inputs) respect point.

        :param ls: (ParameterEntity) length_scale
        :param point: np.array(1xd)
        :param inputs: np.array(nxd)
        :return: np.array(nxdxd)
        """

        derivatives_resp_r = cls.gradient_respect_distance_cross(ls, point, inputs, second=True)

        hessian_respect_point = Distances.gradient_distance_length_scale_respect_point(
            ls.value, point, inputs, second=True
        )

        hess = hessian_respect_point['second']
        part_1 = hess * derivatives_resp_r['first'][0, :][:, np.newaxis, np.newaxis]

        grad = hessian_respect_point['first']
        part_2 = grad[:, :, None] * grad[:, None, :]
        part_2 *= derivatives_resp_r['second'][0, :][:, np.newaxis, np.newaxis]

        hessian = part_1 + part_2


        # hessian = {}
        # for i in xrange(inputs.shape[0]):
        #     hess = hessian_respect_point['second'][i]
        #     part_1 = hess * derivatives_resp_r['first'][0, i]
        #
        #     grad = hessian_respect_point['first'][i:i+1, :]
        #     part_2 = np.dot(grad.transpose(), grad)
        #     part_2 *= derivatives_resp_r['second'][0, i]
        #
        #     hessian[i] = part_1 + part_2

        return hessian
