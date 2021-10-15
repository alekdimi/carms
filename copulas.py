import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod


class Copula(ABC):
    @abstractmethod
    def __init__(self, dim):
        super(Copula, self).__init__()
        self.dim = dim

    @abstractmethod
    def sample(self, shape):
        raise NotImplementedError

    @abstractmethod
    def bivariate_cdf(self, p, q):
        raise NotImplementedError


class Dirichlet(Copula):
    def __init__(self, dim, inverted=False):
        super(Dirichlet, self).__init__(dim=dim)
        self.inverted = inverted


    def sample(self, shape):
        u_iid = tf.random.uniform(tf.concat([(self.dim, ), shape], axis=0), maxval=1.0)
        e = -tf.math.log(u_iid + 1e-8)
        d = e / tf.reduce_sum(e, axis=0, keepdims=True)
        u = tf.pow(1 - d, self.dim - 1)
        return 1 - u if self.inverted else u


    def bivariate_cdf(self, p, q):
        dim = tf.convert_to_tensor(self.dim, dtype=tf.float32)
        if self.inverted:
            term = tf.pow(1 - p, 1 / (dim - 1)) + tf.pow(1 - q, 1 / (dim - 1)) - 1
            return 2 * p - 1 + tf.pow(tf.maximum(term, 0), dim - 1)
        else:
            term = tf.pow(p, 1 / (dim - 1)) + tf.pow(q, 1 / (dim - 1)) - 1
            return tf.pow(tf.maximum(term, 0), dim - 1)


class Gaussian(Copula):
    def __init__(self, dim):
        super(Gaussian, self).__init__(dim=dim)

        mvn_cov = (tf.ones((dim, dim)) - tf.eye(dim) * dim * 1.0001) / (1 - dim)
        self.cholt = tf.transpose(tf.linalg.cholesky(mvn_cov))

        self.univariate_cdf = tfp.distributions.Normal(0., 1.).cdf
        self.univariate_inverse_cdf = tfp.distributions.Normal(0., 1.).quantile
        self.numpy_bivariate_cdf = lambda x: multivariate_normal(mean=[0., 0.], allow_singular=True, 
            cov=np.array([[1.001, 1 / (1 - dim)], [1 / (1 - dim), 1.001]])).cdf(x).astype(np.float32).reshape(x.shape[:-1])


    def sample(self, shape):
        z = tf.random.normal(tf.concat([shape, (self.dim, )], axis=0))
        u = self.univariate_cdf(tf.experimental.numpy.moveaxis(z @ self.cholt, -1, 0))
        return u


    def bivariate_cdf(self, p, q):
        p_inv = tf.clip_by_value(self.univariate_inverse_cdf(p), -10, 10)
        q_inv = tf.clip_by_value(self.univariate_inverse_cdf(q), -10, 10)
        return tf.numpy_function(self.numpy_bivariate_cdf, [tf.stack([p_inv, q_inv], axis=-1)], tf.float32)