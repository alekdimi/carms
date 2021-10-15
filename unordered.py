import tensorflow as tf
from itertools import permutations


def beam_search(marglogp, k, stochastic=False):
    # marglogp should be [batch_size, n_dimensions, n_categories_per_dimension]
    # and should be normalized e.g. marglogp.exp().sum(-1) == 1
    # If stochastic is True, this is stochastic beam search https://arxiv.org/abs/1903.06059
    phi = marglogp[:, 0, :]
    criterium = phi
    if stochastic:
        g_phi, _ = gumbel_with_maximum(phi, tf.zeros(phi.shape[:-1]))
        criterium = g_phi

    crit_topk, ind_topk = tf.math.top_k(criterium, k)

    if stochastic:
        g_phi = crit_topk
        phi = gather_topk(phi, ind_topk)
    else:
        phi = crit_topk

    batch_size = phi.shape[0]
    n_dim = marglogp.shape[1]

    ind_first_action = ind_topk
    trace = []

    # Forward computation
    for i in range(1, n_dim):
        marglogpi = marglogp[:, i, :]

        num_actions = marglogpi.shape[-1]
        # expand_phi = [batch_size, num_parents, num_actions]
        expand_phi = phi[:, :, None] + marglogpi[:, None, :]
        expand_phi_flat = tf.reshape(expand_phi, [batch_size, -1])
        if stochastic:
            expand_g_phi, _ = gumbel_with_maximum(expand_phi, g_phi)
            criterium = tf.reshape(expand_g_phi, [batch_size, -1])
        else:
            criterium = expand_phi_flat

        crit_topk, ind_topk = tf.math.top_k(criterium, k)
        ind_parent, ind_action = ind_topk // num_actions, ind_topk % num_actions

        if stochastic:
            g_phi = crit_topk
            phi = gather_topk(expand_phi_flat, ind_topk)
        else:
            phi = crit_topk

        trace.append((ind_parent, ind_action))

    # Backtrack to get the sample
    prev_ind_parent = None
    actions = []
    for ind_parent, ind_action in reversed(trace):
        if prev_ind_parent is not None:
            # ind_action = tf.compat.v1.batch_gather(ind_action, prev_ind_parent)
            # ind_parent = tf.compat.v1.batch_gather(ind_parent, prev_ind_parent)
            ind_action = tf.gather(ind_action, prev_ind_parent, batch_dims=-1)
            ind_parent = tf.gather(ind_parent, prev_ind_parent, batch_dims=-1)
            
        actions.append(ind_action)
        prev_ind_parent = ind_parent

    if prev_ind_parent is None:
        actions.append
    actions.append(
        # tf.compat.v1.batch_gather(ind_first_action, prev_ind_parent)
        tf.gather(ind_first_action, prev_ind_parent, batch_dims=-1)
        if prev_ind_parent is not None
        else ind_first_action
    )
    return tf.stack(list(reversed(actions)), axis=-1), phi, g_phi if stochastic else None


def compute_log_R_O_nfac(log_p, so_perms):
    """
    Computes all first and second order log ratio's by computing P(S)
    for all second order sets leaving two elements out of S
    where the individual P(S) are computed by naive enumeration of all permutations
    This is inefficient especially for large sample sizes but can be used
    to validate alternative implementations
    """
    k = int(log_p.shape[-1])
    keys, rest = so_perms
    first, second = tf.unstack(keys, axis=-1)

    norm1 = log1mexp(tf.gather(log_p, first, axis=-1))
    norm2 = norm1 + log1mexp(tf.gather(log_p, second, axis=-1) - norm1)

    # Index to get
    # (batch_size, num_second_orders, num_perms, rest=k-2)
    log_p_rest = tf.gather(log_p, rest, axis=-1) - norm2[..., None, None]

    # (batch_size, num_second_orders, num_perms)
    logprobs = log_pl_rec(log_p_rest, -1)

    # (batch_size, num_second_orders)
    log_P = tf.reduce_logsumexp(logprobs, axis=-1)

    # We build the 2d matrix of second order values as a list of list (of batch values)
    # such that we can convert it to a tensor later
    # Probably should also be possible with some scatter functionality
    ind = 0
    log_P2s_list = [[None] * k for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            log_P2_ij = log_P[:, ind]
            log_P2s_list[i][j] = log_P2_ij
            log_P2s_list[j][i] = log_P2_ij
            ind += 1

    # Compute first order log_P
    for i in range(k):
        # P(S) = sum_{s in S} p(s) P^{D\s}(S\s)
        log_p_without_i = tf.concat((log_p[:, :i], log_p[:, i + 1:]), axis=-1) - log1mexp(log_p[:, i, None])
        log_P2s_without_i = tf.stack(log_P2s_list[i][:i] + log_P2s_list[i][i + 1:], axis=-1)
        log_P1_i = tf.reduce_logsumexp(log_p_without_i + log_P2s_without_i, axis=-1)
        log_P2s_list[i][i] = log_P1_i

    log_P2s_list_flat = [log_P2s_list[i][j] for i in range(k) for j in range(k)]
    log_P2s = tf.reshape(tf.stack(log_P2s_list_flat, axis=1), [-1, k, k])
    log_P1s = tf.stack([log_P2s_list[i][i] for i in range(k)], axis=1)

    log_P = tf.reduce_logsumexp(log_p + log_P1s, axis=-1)

    # Bit hacky but if we have (allmost) all probability mass on a few
    # categories we have numerical problems since the probability for other classes
    # is basically zero
    # In this case we can just compute an exact gradient
    # Whereas we can just compute an exact gradient by setting
    # We choose this where the probability mass > 1 - 1e-5, so approx logprob > -1e-5
    is_exact = tf.reduce_logsumexp(log_p, axis=-1) > -1e-5

    log_R1 = log_P1s - log_P[..., None]
    log_R2 = log_P2s - log_P1s[..., None]

    log_R1 = tf.where(tf.broadcast_to(is_exact[:, None], log_R1.shape), tf.zeros_like(log_R1), log_R1)
    log_R2 = tf.where(tf.broadcast_to(is_exact[:, None, None], log_R2.shape), tf.zeros_like(log_R2), log_R2)

    #     log_R1[is_exact] = 0
    #     log_R2[is_exact] = 0

    tf.debugging.check_numerics(log_R1, "Nans in log_R1")
    tf.debugging.check_numerics(log_R2, "Nans in log_R2")

    return log_R1, log_R2


def gather_topk(vals, ind):
    # https://stackoverflow.com/questions/54196149/how-to-use-indices-from-tf-nn-top-k-with-tf-gather-nd
    # This would have been one line in pytorch, should be an easier way?
    inds = tf.meshgrid(*(tf.range(s) for s in ind.shape), indexing='ij')
    # Stack complete index
    index = tf.stack(inds[:-1] + [ind], axis=-1)
    return tf.gather_nd(vals, index)


def gumbel_with_maximum(phi, T, axis=-1):
    g_phi = phi - tf.math.log(-tf.math.log(tf.random.uniform(phi.shape)))
    Z, argmax = tf.reduce_max(g_phi, axis=-1), tf.argmax(g_phi, axis=-1)
    g = shift_gumbel_maximum(g_phi, T, axis, Z=Z)
    return g, argmax


def shift_gumbel_maximum(g_phi, T, axis=-1, Z=None):
    g = _shift_gumbel_maximum(g_phi, T, axis, Z)
    g_inv = _shift_gumbel_maximum(g, Z, axis)

    CHECK_VALIDITY = True
    if CHECK_VALIDITY:
        check = tf.reduce_all(((g_phi - g_inv) < 1e-3) | (g_phi == g_inv))
        # print("Check", check)#tf.assert(check)

    return g


def _shift_gumbel_maximum(g_phi, T, axis=-1, Z=None):
    if Z is None:
        Z = tf.reduce_max(g_phi, axis=axis)
    T_ = tf.expand_dims(T, axis=axis)
    u = T_ - g_phi + tf.math.log1p(-tf.exp(g_phi - tf.expand_dims(Z, axis=axis)))
    return T_ - tf.nn.relu(u) - tf.nn.softplus(-tf.abs(u))


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -tf.abs(x)
    return tf.where(x > -0.693, tf.math.log(-tf.math.expm1(x)), tf.math.log1p(-tf.exp(x)))


def all_2nd_order_perms(k):
    ap = tf.constant(list(permutations(range(k)))) # all_perms(k)
    apf = tf.reshape(tf.boolean_mask(ap, ap[:, 0] < ap[:, 1]), [k * (k - 1) // 2, -1, k])
    return apf[:, 0, :2], apf[:, :, 2:]


def log_pl_rec(log_p, dim=-1):
    """Recursive function of Plackett Luce log probability has better numerical stability
    since 1 - sum_i p_i can get very close to 0, this version never computes sum p_i directly"""
    assert dim == -1
    if log_p.shape[-1] == 1:
        return log_p[..., 0]
    return log_p[..., 0] + log_pl_rec(log_p[..., 1:] - log1mexp(log_p[..., 0:1]), dim=dim)