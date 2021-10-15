import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import copulas
import unordered
import antithetic


class CategoricalVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, prior_logits, encoder_opt, decoder_opt, prior_opt, grad_type,
                 num_samples, num_eval_samples, inverted=False, num_mc=100, theta_opt=None, control_nn=None, mle=False): 
        super().__init__('cvae')
        
        self.mle = mle
        self.encoder = encoder
        self.decoder = decoder
        self.prior_logits = prior_logits
        self.grad_type = grad_type.lower()
        self.inverted = inverted
        self.num_mc = num_mc
        
        self.num_latent = tf.shape(prior_logits)[0]
        self.num_categories = tf.shape(prior_logits)[1]
        
        if self.grad_type == 'arsm':
            self.num_samples = np.maximum(1, num_samples // (self.num_categories * (self.num_categories - 1) // 2))
            self.swaps = self._get_arsm_swaps(self.num_categories)

        elif self.grad_type == 'unord':
            self.num_samples = np.minimum(7, np.minimum(self.num_categories, num_samples))
            self.cache = unordered.all_2nd_order_perms(self.num_samples)

        elif 'carms' in self.grad_type:
            self.num_samples = num_samples
            copula = copulas.Dirichlet(self.num_samples, inverted=self.inverted)
            if self.grad_type == 'ecarms':
                self.cat_copula = antithetic.GumbelCategorical(self.num_categories, copula, num_mc=num_mc)
            elif self.grad_type == 'carms':
                self.cat_copula = antithetic.InverseCategorical(self.num_categories, copula, weighted=False)
            elif self.grad_type == 'sbcarms':
                self.cat_copula = antithetic.StickBreakingCategorical(self.num_categories, copula)
            else:
                raise NotImplementedError

        else:
            self.num_samples = num_samples
        self.num_eval_samples = num_eval_samples 
        
        self.prior_dist  = tfp.distributions.OneHotCategorical(logits=self.prior_logits)
        self.prior_opt   = prior_opt
        self.encoder_opt = encoder_opt
        self.decoder_opt = decoder_opt
        
    
    def _get_arsm_swaps(self, k, dtype=np.float32, tensor=True):
        swaps = np.tile(np.expand_dims(np.eye(k, dtype=dtype), axis=(0, 1)), [k, k, 1, 1])
        for i in range(k):
            for j in range(k):
                swaps[i, j, [i, j]] = swaps[i, j, [j, i]]
        return tf.convert_to_tensor(swaps) if tensor else swaps


    def _get_avg_num_evals(self, fz):
        num_evals = tf.map_fn(lambda x: tf.cast(tf.size(tf.unique(x)[0]), tf.float32), fz)
        self._std_evals = tf.math.reduce_variance(num_evals)
        return tf.reduce_mean(num_evals)

    
    def matrix_grad(self, ohz, fz, p, joint=None, ratio=None):
        if ratio is None:
            if joint is not None:
                ratio = tf.clip_by_value((tf.expand_dims(p, -1) @ tf.expand_dims(p, -2) + 1e-3) / (joint + 1e-3), 0.01, 10)
            else:
                ratio = tf.ones(tf.concat([tf.shape(p), (tf.shape(p)[-1], )], axis=0))
            
        M          = tf.shape(ohz)[0] 
        ratio_ndmm = tf.transpose(ohz, (1, 2, 0, 3)) @ ratio @ tf.transpose(ohz, (1, 2, 3, 0))
        I          = tf.expand_dims(tf.expand_dims(tf.eye(M), axis=0), axis=0)
        offdiag    = ratio_ndmm * (tf.ones(tf.shape(ratio_ndmm)) - I)
        diag       = tf.reduce_sum(offdiag, axis=-1, keepdims=True) * I
        R          = (diag - offdiag) / tf.cast(M, tf.float32)
        score      = ohz - tf.expand_dims(p, axis=0)
        final      = tf.transpose(R @ tf.expand_dims(tf.expand_dims(tf.transpose(fz), axis=1), axis=-1), (2, 0, 1, 3))
        grads      = tf.squeeze(tf.expand_dims(score, axis=-1) @ tf.expand_dims(final, axis=-1), axis=-1) * tf.cast(M / (M - 1), tf.float32)
        return grads

    
    def sample_latent(self, logits, num_samples=None, evaluate=False, dtype=None):
        if num_samples is None:
            num_samples = self.num_samples
        if evaluate or 'carms' not in self.grad_type:
            z = tfp.distributions.OneHotCategorical(logits=logits).sample(num_samples)
        elif 'carms' in self.grad_type:
            probs  = tf.math.softmax(logits, axis=-1)
            z = self.cat_copula.sample(probs)
        else:
            raise NotImplementedError
        return tf.cast(z, dtype) if dtype is not None else z
        
    
    def call(self, input_batch, num_samples=None, evaluate=False, z=None):
        if num_samples is None:
                num_samples = self.num_samples

        if self.mle:
            top, bottom = tf.split(input_batch, 2, axis=1)
            input_tensor = tf.tile(top, (1, 2))
            output_tensor = tf.tile(bottom, (1, 2))
        else:
            input_tensor, output_tensor = input_batch, input_batch

        logits = self.encoder(input_tensor)
        if z is not None:
            tf.debugging.assert_equal(tf.shape(z)[1:], tf.shape(logits))
        else:    
            z = self.sample_latent(logits, num_samples, evaluate)

        z_flat = tf.reshape(z, tf.concat([tf.shape(z)[:-2], [tf.reduce_prod(tf.shape(self.prior_logits))]], axis=0))
        recon_logits = self.decoder(z_flat)
        bernoulli = tfp.distributions.Bernoulli(logits=recon_logits)

        log_decoder = tf.reduce_sum(bernoulli.log_prob(tf.expand_dims(output_tensor, axis=0)), axis=-1)
        if self.mle:
            return log_decoder if evaluate else (log_decoder, logits, tf.cast(z, tf.float32))

        log_encoder = tf.reduce_sum(tfp.distributions.OneHotCategorical(logits=logits).log_prob(z), axis=-1)
        log_prior   = tf.reduce_sum(self.prior_dist.log_prob(z), axis=-1)
        elbo = log_decoder + log_prior - log_encoder
        return elbo if evaluate else (elbo, logits, tf.cast(z, tf.float32))

    
    def get_logits_gradients(self, logits, z, elbo, input_batch, grad_type=None, count_evals=False):
        if grad_type is None:
            grad_type = self.grad_type
        probs = tf.math.softmax(logits, axis=-1)    
        
        if grad_type == 'loorf' or 'carms' in grad_type:
            if grad_type == 'sbcarms':
                ratio = self.cat_copula.bivariate_ratio(probs)
                grads = self.matrix_grad(z, elbo, probs, None, ratio=ratio)
            else:
                joint = self.cat_copula.bivariate_pmf(probs) if 'carms' in grad_type else None
                grads = self.matrix_grad(z, elbo, probs, joint)
            num_evals = self._get_avg_num_evals(tf.transpose(elbo)) if count_evals else -1.
            
        elif grad_type == 'arsm':
            assert len(tf.shape(logits)) == 3
            N, D, C = tf.shape(logits)[0], tf.shape(logits)[1], tf.shape(logits)[2]
            pi     = tfp.distributions.Dirichlet(tf.ones(C)).sample((N, D))
            log_pi = tf.expand_dims(tf.expand_dims(tf.math.log(1e-8 + pi), axis=0), axis=0) @ tf.expand_dims(self.swaps, axis=2)
            z      = tf.one_hot(tf.math.argmin(log_pi - tf.expand_dims(tf.expand_dims(logits, axis=0), axis=0), axis=-1), C)
            fz     = tf.reshape(self.call(input_batch, z=tf.reshape(z, [-1, N, D, C]), evaluate=True), (C, C, N))    
            fnorm  = tf.expand_dims(fz - tf.reduce_mean(fz, axis=0, keepdims=True), axis=2)
            grads  = tf.transpose(fnorm * (1 - tf.cast(C, tf.float32) * tf.expand_dims(tf.transpose(pi), axis=0)), perm=(1, 3, 2, 0))
            num_evals = self._get_avg_num_evals(tf.reshape(tf.transpose(fz), (N, C * C))) if count_evals else -1.
            
        elif grad_type == 'unord':       
            sample, log_p_sample, _ = unordered.beam_search(tf.nn.log_softmax(logits, axis=-1), self.num_samples, stochastic=True)
            z_beam = tf.one_hot(tf.transpose(sample, (1, 0, 2)), self.num_categories)
            neg_elbo_samples = tf.transpose(self.call(input_batch, z=z_beam, evaluate=True))
            log_R1, log_R2 = unordered.compute_log_R_O_nfac(log_p_sample, self.cache)
            bl_vals = tf.reduce_sum(tf.exp(log_p_sample[:, None, :] + log_R2) * neg_elbo_samples[:, None, :], axis=-1)
            adv = neg_elbo_samples - bl_vals
            nograd = tf.stop_gradient(tf.transpose(tf.exp(log_R1 + log_p_sample) * adv))
            grads = tf.expand_dims(tf.expand_dims(nograd, -1), -1) * (z_beam - tf.expand_dims(probs, 0)) * self.num_samples
            num_evals = self._get_avg_num_evals(neg_elbo_samples) if count_evals else -1.

        else:
            raise NotImplementedError
            
        grads = tf.cast(grads, tf.float32)
        gradient = tf.reduce_mean(grads, axis=0)
        variance = tf.math.reduce_variance(grads, axis=0)
        return gradient, tf.reduce_mean(variance), num_evals
    

    @tf.function
    def train_step(self, input_batch, count_evals=False):
        if self.grad_type == 'unord':
            with tf.GradientTape(persistent=True) as tape:
                elbo, logits, z = self.call(input_batch)
                loss = -1. * tf.reduce_mean(elbo)
                sample, log_p_sample, _ = unordered.beam_search(tf.nn.log_softmax(logits, axis=-1), 
                                                                self.num_samples, stochastic=True)
                z_beam = tf.one_hot(tf.transpose(sample, (1, 0, 2)), self.num_categories)
                neg_elbo_samples = -1. * tf.transpose(self.call(input_batch, z=z_beam, evaluate=True))
                log_R1, log_R2 = unordered.compute_log_R_O_nfac(log_p_sample, self.cache)
                bl_vals = tf.reduce_sum(tf.exp(log_p_sample[:, None, :] + log_R2) * neg_elbo_samples[:, None, :], axis=-1)
                adv = neg_elbo_samples - bl_vals
                batch_rf_losses = tf.reduce_sum(log_p_sample * tf.stop_gradient(tf.exp(log_R1 + log_p_sample) * adv), -1)
                loss_encoder = tf.reduce_mean(batch_rf_losses)
                
            encoder_grads = tape.gradient(loss_encoder, self.encoder.trainable_variables)
            _, grad_var, num_evals = self.get_logits_gradients(logits, None, None, input_batch, count_evals=count_evals)
            
        else:
            with tf.GradientTape(persistent=True) as tape:
                elbo, logits, z = self.call(input_batch)
                loss = -1. * tf.reduce_mean(elbo)
            grad_logits, grad_var, num_evals = self.get_logits_gradients(logits, z, elbo, input_batch, count_evals=count_evals)
            encoder_grads = tape.gradient(logits, self.encoder.trainable_variables, output_gradients=-1. * grad_logits)

        decoder_grads = tape.gradient(loss, self.decoder.trainable_variables)
        if not self.mle:
            prior_grads = tape.gradient(loss, self.prior_dist.trainable_variables)
        del tape

        self.encoder_opt.apply_gradients(list(zip(encoder_grads, self.encoder.trainable_variables)))
        self.decoder_opt.apply_gradients(list(zip(decoder_grads, self.decoder.trainable_variables)))
        if not self.mle:
            self.prior_opt.apply_gradients(list(zip(prior_grads, self.prior_dist.trainable_variables)))
        return loss, grad_var, num_evals
    
    @tf.function
    def evaluate(self, dataset, process_batch_input, max_step=1000):
        loss, n = 0., 0.
        for input_batch in dataset.map(process_batch_input):
            if n >= max_step: # because train_ds is a 'repeat' dataset
                break
            elbo = self.call(input_batch, num_samples=self.num_eval_samples, evaluate=True)
            bound = tf.reduce_logsumexp(elbo, axis=0, keepdims=False) - tf.math.log(1e-8 + tf.cast(tf.shape(elbo)[0], tf.float32))
            loss -= tf.reduce_mean(bound)
            n += 1.
        return loss / n