import os
import tensorflow as tf

import vae
import datasets
import networks


flags.DEFINE_enum('dataset', 'dynamic_mnist', ['static_mnist', 'dynamic_mnist', 'fashion_mnist', 'omniglot'], 'Dataset to use.')
flags.DEFINE_string('logdir', 'logs', 'Directory for storing logs.')
flags.DEFINE_string('ckptdir', 'ckpts', 'Directory for storing checkpoints.')
flags.DEFINE_string('grad_type', 'carms', 'Choice supported: loorf, carms, unord, arsm.')
flags.DEFINE_string('encoder_type', 'linear', 'Choice supported: linear, nonlinear')
flags.DEFINE_integer('num_samples', 7, 'Number of samples for all gradient estimators.')
flags.DEFINE_integer('num_latent', 20, 'Number of latent categorical variables.')
flags.DEFINE_integer('num_categories', 10, 'Number of categories of each latent variable.')
flags.DEFINE_integer('num_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('num_eval_samples', 20, 'Number of samples for evaluation.')
flags.DEFINE_integer('repeat_idx', 0, 'Dummy flag to label the experiments in repeats.')
flags.DEFINE_integer('batch_size', 50, 'Training batch size.')
flags.DEFINE_integer('seed', 1, 'Global random seed.')
flags.DEFINE_integer('summary_every', 1000, 'Global random seed.')
flags.DEFINE_integer('ckpt_every', 10000, 'Global random seed.')
flags.DEFINE_float('encoder_lr', 1e-4, 'Learning rate for encoder (inference) network.')
flags.DEFINE_float('decoder_lr', 1e-4, 'Learning rate for decoder (generation) network.')
flags.DEFINE_float('prior_lr', 1e-2, 'Learning rate for prior variables.')
flags.DEFINE_bool('empirical', True, 'carms will use an empirical bivariate pmf.')
flags.DEFINE_bool('eager', False, 'Enable eager execution.')
flags.DEFINE_bool('demean_input', True, 'Demean for encoder and decoder inputs.')
flags.DEFINE_bool('initialize_with_bias', True, 'Initialize the final layer bias of decoder with dataset mean.')
flags.DEFINE_bool('debug', False, 'Turn on debugging mode.')
flags.DEFINE_bool('mle', False, 'Train an MLE image completion network (True) or a VAE (False).')
flags.DEFINE_bool('evaluate_only', False, 'Load latest checkpoint and evaluate.')


def main(_):
    FLAGS = flags.FLAGS
    tf.random.set_seed(FLAGS.seed)
    if FLAGS.eager:
        tf.config.experimental_run_functions_eagerly(FLAGS.eager)

    if FLAGS.dataset == 'static_mnist':
        train_ds, _, test_ds = datasets.get_static_mnist_batch(FLAGS.batch_size)
        train_size = 50000
    elif FLAGS.dataset == 'dynamic_mnist':
        train_ds, _, test_ds = datasets.get_dynamic_mnist_batch(FLAGS.batch_size)
        train_size = 50000
    elif FLAGS.dataset == 'fashion_mnist':
        train_ds, _, test_ds = datasets.get_dynamic_mnist_batch(FLAGS.batch_size, fashion_mnist=True)
        train_size = 50000
    elif FLAGS.dataset == 'omniglot':
        train_ds, _, test_ds = datasets.get_omniglot_batch(FLAGS.batch_size)
        train_size = 23000

    num_steps_per_epoch = int(train_size / FLAGS.batch_size)
    train_ds_mean = datasets.get_mean_from_iterator(train_ds, dataset_size=train_size, batch_size=FLAGS.batch_size)

    if FLAGS.initialize_with_bias:
        bias_value = -tf.math.log(1./tf.clip_by_value(train_ds_mean, 0.001, 0.999) - 1.)
        bias_initializer = tf.keras.initializers.Constant(bias_value)
    else:
        bias_initializer = 'zeros'

    def process_batch_input(input_batch):
        return tf.cast(tf.reshape(input_batch, [tf.shape(input_batch)[0], -1]), tf.float32)

    if FLAGS.mle:
        encoder_hidden_sizes = [200, FLAGS.num_latent]
        encoder_activations = [tf.keras.layers.LeakyReLU(alpha=0.3), 'linear']
        decoder_hidden_sizes = [200, 784]
        decoder_activations = [tf.keras.layers.LeakyReLU(alpha=0.3), 'linear']
    else:
        if FLAGS.encoder_type == 'linear':
            encoder_hidden_sizes = [FLAGS.num_latent]
            encoder_activations = ['linear']
            decoder_hidden_sizes = [784]
            decoder_activations = ['linear']
        elif FLAGS.encoder_type == 'nonlinear':
            encoder_hidden_sizes = [200, 200, FLAGS.num_latent]
            encoder_activations = [tf.keras.layers.LeakyReLU(alpha=0.3), tf.keras.layers.LeakyReLU(alpha=0.3), 'linear']
            decoder_hidden_sizes = [200, 200, 784]
            decoder_activations = [tf.keras.layers.LeakyReLU(alpha=0.3), tf.keras.layers.LeakyReLU(alpha=0.3), 'linear']
        else:
            raise NotImplementedError

    encoder = networks.Encoder(FLAGS.num_categories, encoder_hidden_sizes, encoder_activations, 
                               mean_xs=train_ds_mean, demean_input=FLAGS.demean_input, name='encoder')
    decoder = networks.Decoder(decoder_hidden_sizes, decoder_activations, demean_input=FLAGS.demean_input, 
                               final_layer_bias_initializer=bias_initializer, name='decoder')
    prior_logits = tf.Variable(tf.zeros([FLAGS.num_latent, FLAGS.num_categories], tf.float32))

    encoder_opt  = tf.keras.optimizers.Adam(learning_rate=tf.constant(FLAGS.encoder_lr))
    decoder_opt  = tf.keras.optimizers.Adam(learning_rate=tf.constant(FLAGS.decoder_lr))
    prior_opt  = tf.keras.optimizers.SGD(learning_rate=tf.constant(FLAGS.prior_lr))

    if FLAGS.grad_type.lower() == 'relax':
        theta_opt = tf.keras.optimizers.Adam(learning_rate=tf.constant(FLAGS.encoder_lr), beta_1=0.999)
        control_network = tf.keras.Sequential()
        control_network.add(tf.keras.layers.Dense(137, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        control_network.add(tf.keras.layers.Dense(FLAGS.num_categories))
    else:
        theta_opt = None
        control_network = None

    cvae = vae.CategoricalVAE(encoder, decoder, prior_logits, encoder_opt, decoder_opt, prior_opt, grad_type=FLAGS.grad_type,
                              num_samples=FLAGS.num_samples, num_eval_samples=FLAGS.num_eval_samples,
                              theta_opt=theta_opt, control_nn=control_network, mle=FLAGS.mle)
    cvae.build(input_shape=(None, 784))
    
    ckptdir = os.path.join(FLAGS.ckptdir, FLAGS.dataset, FLAGS.encoder_type, str(FLAGS.num_samples), str(FLAGS.repeat_idx), FLAGS.grad_type)
    logdir  = os.path.join(FLAGS.logdir, FLAGS.dataset, FLAGS.encoder_type, str(FLAGS.num_samples), str(FLAGS.repeat_idx), FLAGS.grad_type)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    tensorboard_file_writer = tf.summary.create_file_writer(logdir)

    ckpt = tf.train.Checkpoint(cvae=cvae, encoder_opt=cvae.encoder_opt, decoder_opt=cvae.decoder_opt, prior_opt=cvae.prior_opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckptdir, max_to_keep=1)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logging.info('Last checkpoint was restored: %s.', ckpt_manager.latest_checkpoint)
    else:
        logging.info('No checkpoint to load.')

    start_step = encoder_opt.iterations.numpy()
    logging.info('Training starting from step: %s', start_step)
    logging.info('Number of categories: %s', cvae.num_categories)
    logging.info('Number of latent variables: %s', cvae.num_latent)
    logging.info('Number of samples: %s', cvae.num_samples)

    if FLAGS.evaluate_only:
        results_path = os.path.join(FLAGS.ckptdir, 'results.txt')
        logging.info('Writing to %s.' % results_path)
        metrics = {'eval_metric/train': cvae.evaluate(train_ds, process_batch_input, max_step=num_steps_per_epoch),
                    'eval_metric/test': cvae.evaluate(test_ds, process_batch_input)}
        with open(results_path, 'a') as f:
            f.write(ckptdir + '\t' + '\t'.join([k + ': ' + str(v.numpy() / (1 + int(FLAGS.mle))) for k, v in metrics.items()]) + '\n')
        return None

    tf.debugging.enable_check_numerics()
    train_iter = train_ds.__iter__()
    varss, var_idx = [], 0.
    for step in range(start_step, FLAGS.num_steps + 1):
        input_batch = process_batch_input(train_iter.next())
        train_loss, grad_var, num_evals = cvae.train_step(input_batch, tf.cast(step % FLAGS.summary_every == 0, tf.bool))

        varss.append(grad_var)
        var_idx += 1

        if step % FLAGS.summary_every == 0:
            grad_var = tf.math.add_n(varss) / var_idx
            varss, var_idx = [], 0.
            
            metrics = {'train_objective': train_loss,
                       'eval_metric/train': cvae.evaluate(train_ds, process_batch_input, max_step=num_steps_per_epoch),
                       'eval_metric/test': cvae.evaluate(test_ds, process_batch_input),
                       'var/grad': grad_var,
                       'num/evals': num_evals}
            tf.print(step, metrics)
            with tensorboard_file_writer.as_default():
                for k, v in metrics.items():
                    tf.summary.scalar(k, v, step=step)

        if step % FLAGS.ckpt_every == 0 and step > 0:
            ckpt_save_path = ckpt_manager.save()
            logging.info('Saving checkpoint for step %d at %s.', step, ckpt_save_path)


if __name__ == '__main__':
    app.run(main)