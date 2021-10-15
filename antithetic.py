import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC, abstractmethod
from tensorflow.experimental.numpy import moveaxis


class Categorical(ABC):
    @abstractmethod
    def __init__(self, num_categories, copula):
        super(Categorical, self).__init__()
        self.num_categories = num_categories
        self.copula = copula
        

    @abstractmethod
    def sample(self, p, num_samples):
        raise NotImplementedError

    @abstractmethod
    def bivariate_pmf(self, p):
        raise NotImplementedError
        
        
class StickBreakingCategorical(Categorical):
    def __init__(self, num_categories, copula):
        super(StickBreakingCategorical, self).__init__(num_categories=num_categories, copula=copula)
        

    def sample(self, p, num_samples=1):
        tf.debugging.assert_equal(tf.shape(p)[-1], self.num_categories)
        shape = tf.concat([(num_samples, ), tf.shape(p)], axis=0)
        argsort = tf.gather(tf.eye(self.num_categories), tf.argsort(p, axis=-1))
        p = tf.squeeze(tf.linalg.matmul(argsort, tf.expand_dims(p, axis=-1)), axis=-1)
        pi = p / tf.cumsum(p, axis=-1, reverse=True)
        u = self.copula.sample(shape)
        z = tf.argmax(u < pi, axis=-1)
        ohz = tf.one_hot(z, shape[-1])
        ohz = tf.squeeze(tf.linalg.matmul(argsort, tf.expand_dims(ohz, axis=-1), transpose_a=True), axis=-1)
        return tf.squeeze(ohz, axis=1) if num_samples == 1 else ohz

    def bivariate_ratio(self, p):
        C = self.num_categories
        tf.debugging.assert_equal(tf.shape(p)[-1], C)
        
        argsort = tf.gather(tf.eye(C), tf.argsort(p, axis=-1))
        p_sorted = tf.squeeze(tf.linalg.matmul(argsort, tf.expand_dims(p, axis=-1)), axis=-1)
        pi = p_sorted / tf.cumsum(p_sorted, axis=-1, reverse=True)

        rows, cols = tf.meshgrid(tf.range(C), tf.range(C))
        Pi = tf.expand_dims(tf.expand_dims(pi, -2), -2)
        Phi = self.copula.bivariate_cdf(Pi, Pi) # tf.maximum(0, 2 * Pi - 1)        

        rcmin = tf.expand_dims(tf.minimum(rows, cols), -1)
        ranged = tf.reshape(tf.range(C), (1, 1, C))

        idx_shape = tf.concat((tf.ones(len(tf.shape(pi)) - 1, dtype=tf.int32), [C, C, C]), axis=0)
        idx1 = tf.reshape(tf.cast(ranged < rcmin, tf.float32), idx_shape)
        idx2 = tf.reshape(tf.cast(ranged == rcmin, tf.float32), idx_shape)

        prod1 = tf.reduce_prod(idx1 * (tf.square(1 - Pi) / (1 - 2 * Pi + Phi + 1e-8)) + (1 - idx1), axis=-1)
        prod2 = tf.reduce_prod(idx2 * (1 - Pi) * Pi / (Pi - Phi + 1e-6) + (1 - idx2), axis=-1)

        final = (prod1 * prod2) * (1 - tf.eye(C))
        ratio = tf.linalg.matmul(tf.linalg.matmul(argsort, final), argsort, transpose_b=True)
        return ratio


    def bivariate_pmf(self, p):
        ratio = self.bivariate_ratio(p)
        indep = ((tf.expand_dims(p, -1) @ tf.expand_dims(p, -2)))   
        joint = ((1e-6 + indep) / (1e-6 + ratio)) * (1 - tf.eye(C)) #- 1e6 * tf.eye(C)
        return joint + 1e-6 * tf.ones(tf.shape(joint))


class InverseCategorical(Categorical):
    def __init__(self, num_categories, copula, weighted=False):
        super(InverseCategorical, self).__init__(num_categories=num_categories, copula=copula)

        self.weighted = weighted
        self.permutations = self._get_permutations()
        

    def sample(self, p, num_samples=1):
        tf.debugging.assert_equal(tf.shape(p)[-1], self.num_categories)
        shape = tf.concat([(num_samples, ), tf.shape(p)[:-1]], axis=0)
        if self.weighted:
            log_weights  = tf.math.log(1e-8 + tf.transpose(self._get_weights(p), (1, 2, 0))) 
            ohords       = tf.cast(tfp.distributions.OneHotCategorical(logits=log_weights).sample(num_samples), tf.float32)
            exp_ohords   = tf.expand_dims(tf.expand_dims(tf.expand_dims(ohords, -2), -2), -2)
            exp_perms    = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.transpose(self.permutations, (1, 2, 0)), 0), 0), 0), -1)
            permutations = tf.squeeze(exp_ohords @ exp_perms, axis=(-1, -2))
        else:
            indices      = tf.reshape(tf.random.categorical(tf.zeros([1, tf.shape(self.permutations)[0]]), tf.reduce_prod(shape)), shape)   
            permutations = tf.gather(self.permutations, indices)

        p_shuffled   = tf.squeeze(permutations @ tf.expand_dims(tf.expand_dims(p, axis=0), axis=-1), axis=-1)
        left, right  = self._get_boundaries(p_shuffled)
        u            = tf.expand_dims(self.copula.sample(shape), axis=-1)
        ohz_shuffled = tf.cast(tf.math.logical_and(tf.expand_dims(left, 0) < u, u <= tf.expand_dims(right, 0)), dtype=tf.float32)
        inverse_perm = tf.expand_dims(tf.linalg.matrix_transpose(permutations), axis=0)
        ohz          = tf.squeeze(inverse_perm @ tf.expand_dims(ohz_shuffled, axis=-1), axis=-1)
        return tf.squeeze(ohz, axis=1) if num_samples == 1 else ohz
    

    def bivariate_pmf(self, p):
        tf.debugging.assert_equal(tf.shape(p)[-1], self.num_categories)
        perms       = tf.expand_dims(tf.expand_dims(self.permutations, axis=1), axis=1)
        p_swapped   = tf.squeeze(perms @ tf.expand_dims(tf.expand_dims(p, 0), -1), axis=-1)
        left, right = self._get_boundaries(p_swapped)
        L           = tf.tile(tf.expand_dims(left, axis=-2), tf.concat((tf.repeat(1, tf.rank(left) - 1), (self.num_categories, 1)), axis=0))
        R           = tf.tile(tf.expand_dims(right, axis=-2), tf.concat((tf.repeat(1, tf.rank(right) - 1), (self.num_categories, 1)), axis=0))
        LT, RT      = tf.linalg.matrix_transpose(L), tf.linalg.matrix_transpose(R)
        bicdf       = self.copula.bivariate_cdf
        intervals   = bicdf(L, LT) + bicdf(R, RT) - bicdf(L, RT) - bicdf(R, LT)  
        joint       = tf.linalg.matrix_transpose(perms) @ intervals @ perms
        if self.weighted:
            weights = tf.expand_dims(tf.expand_dims(self._get_weights(p), -1), -1)
            return tf.reduce_sum(joint * weights, axis=0)
        return tf.reduce_mean(joint, axis=0)  
    
    
    def _get_weights(self, p):
        pp = tf.squeeze(tf.expand_dims(tf.expand_dims(self.permutations, 1), 1) @ tf.expand_dims(tf.expand_dims(p, -1), 0), axis=-1)
        w = tf.transpose(tf.transpose(pp)[0] * tf.transpose(pp)[-1])
        return w / tf.reduce_sum(w, axis=0, keepdims=True)
    

    def _get_boundaries(self, p):
        left  = tf.math.cumsum(tf.concat([tf.zeros(tf.concat([tf.shape(p)[:-1], (1, )], axis=0)), p], axis=-1), axis=-1)
        left  = tf.transpose(tf.transpose(left)[:-1])
        right = tf.math.cumsum(p, axis=-1)
        return left, right
    
    
    def _get_permutations(self, dtype=np.float32, tensor=True):
        k, num_cat = 0, self.num_categories
        perms = np.tile(np.expand_dims(np.eye(num_cat, dtype=dtype), 0), [num_cat * (num_cat - 1), 1, 1])
        for i in range(num_cat):
            for j in range(num_cat - 1):
                perms[k] = np.roll(perms[k], i, axis=0)
                perms[k, 1:] = np.roll(perms[k, 1:], j, axis=0)
                k += 1
        argmax = np.argmax(perms, axis=-1)
        perms = perms[argmax.T[0] < argmax.T[-1]]
        return tf.convert_to_tensor(perms) if tensor else perms
    
        
class GumbelCategorical(Categorical):
    def __init__(self, num_categories, copula, num_mc=100):
        super(GumbelCategorical, self).__init__(num_categories=num_categories, copula=copula)
        self.num_mc = num_mc

            
    def sample(self, p, num_samples=1):
        tf.debugging.assert_equal(tf.shape(p)[-1], self.num_categories)
        u = self.copula.sample(tf.concat([(num_samples, ), tf.shape(p)], axis=0))
        z = tf.argmax(tf.math.log(1e-8 + p) -tf.math.log(-tf.math.log(1e-8 + u) + 1e-8), axis=-1)
        ohz = tf.one_hot(z, tf.shape(p)[-1])
        if num_samples == 1:
            return tf.squeeze(ohz, axis=1)
        return ohz


    def bivariate_pmf(self, p):
        tf.debugging.assert_equal(tf.shape(p)[-1], self.num_categories)
        dim = self.copula.dim
        ohz = self.sample(p, max(2, self.num_mc))
        joint = tf.reduce_mean(tf.experimental.numpy.moveaxis(ohz, 0, -1) @ (tf.ones((dim, dim)) - tf.eye(dim)) \
                                    @ tf.experimental.numpy.moveaxis(ohz, 0, -2), axis=0) / (dim * (dim - 1))
        return 0.5 * (joint + tf.linalg.matrix_transpose(joint))
