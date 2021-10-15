import vae

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
tf.random.set_seed(1)

M, N, D, C = 3, 1000, 3, 3
array = [1, 10, 100, 1000]

methods = ['carms', 'ecarms', 'unord', 'loorf']
method_names = {'carms': 'CARMS-I', 'ecarms': 'CARMS-G', 'unord': 'UNORD', 'loorf': 'LOORF'}

cmap = plt.get_cmap('Dark2')
colors = [cmap(0), cmap(4), cmap(2), cmap(1)]
fontsize = 11

fmat = 1. * tf.reshape(tf.range(1, D * C + 1, dtype=tf.float32), (D, C)) * C
def f(ohz, const=1):
    return tf.reduce_sum(tf.broadcast_to(fmat, tf.shape(ohz)) * tf.cast(ohz, tf.float32) / const , axis=(-1, -2))

fig, ax = plt.subplots(1, len(array), figsize=(17, 2.5))
for ii, K in enumerate(array):
    dist = tfp.distributions.Dirichlet(C * [K])
    probs = tf.tile(tf.reshape(dist.sample(D), (1, D, C)), (N, 1, 1))
    logits = tf.math.log(probs)
    cvae, grads, means, varss, eval_mean, eval_std = {}, {}, {}, {}, {}, {}
    for method in methods:
        cvae[method] = vae.CategoricalVAE(None, None, tf.zeros([D, C]), None, None, None, method, M, None, inverted=False, num_mc=200)
        z = cvae[method].sample_latent(logits, dtype=tf.float32)    
        cvae[method].call = lambda input_batch, z, evaluate: f(z)
        elbo = cvae[method].call(None, z, True)
        
        grad, var, evm = cvae[method].get_logits_gradients(logits, z, elbo, None, None, True)
        evs = cvae[method]._std_evals
        grads[method] = grad.numpy()
        means[method] = tf.reduce_mean(grads[method], axis=0).numpy()
        varss[method] = tf.math.reduce_variance(grads[method], axis=0).numpy()
        eval_mean[method] = evm.numpy()
        eval_std[method] = evs.numpy()
        
    labels, data = [], []
    for method, var in varss.items():
        var = np.log(var + 1e-12)
        labels.append(method_names[method] + '\n' + str(np.median(var).round(2)) + '±' + str((np.std(var)).round(2)))
        data.append(var.flatten()) 
    bp = ax[ii].boxplot(data, labels=labels, patch_artist=True)
    ax[ii].set_title('α=' + str(K) + ', H(π)=' + str(dist.entropy().numpy().round(2)))
    for spine in ['left', 'right', 'top', 'bottom']:
        ax[ii].spines[spine].set_visible(False)   
    for item in ([ax[ii].title, ax[ii].xaxis.label, ax[ii].yaxis.label] + \
                 ax[ii].get_xticklabels() +  ax[ii].get_yticklabels()):
        item.set_fontsize(fontsize)
        #item.set_fontweight('bold')
    ax[ii].title.set_fontsize(fontsize + 3)
    for median in bp['medians']:
        median.set(color='k', linewidth=3)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

fig.tight_layout()
plt.show()
