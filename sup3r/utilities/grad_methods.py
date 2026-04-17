"""Multi-task gradient methods for TensorFlow.

Implements gradient surgery / reweighting algorithms for training with
multiple loss terms:

- **PCGrad** (Yu et al., *"Gradient Surgery for Multi-Task Learning"*,
  NeurIPS 2020): projects out conflicting gradient components.
- **MGDA-II** (Désidéri, *"Multiple-gradient descent algorithm (MGDA)
  for multiobjective optimization"*, C. R. Acad. Sci. Paris 2012):
  direct active-set QP solver on the Gram matrix — finds the
  minimum-norm point in the convex hull of per-task gradients in at
  most *T* iterations.
"""

import tensorflow as tf


def _replace_none_grads(grads, ref_grads):
    """Replace None gradients with zeros matching the variable shape.

    Parameters
    ----------
    grads : list
        List of gradient tensors, some of which may be None.
    ref_grads : list
        Reference list of tensors (e.g. training weights) with correct
        shapes to use for creating zeros when a gradient is None.
    """
    return [
        tf.zeros_like(r) if g is None else g for g, r in zip(grads, ref_grads)
    ]


def _flatten_grads(grads):
    """Concatenate a list of per-variable gradient tensors into one vector."""
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)


def _unflatten_grads(flat, shapes):
    """Split a flat gradient vector back into per-variable tensors."""
    results = []
    offset = 0
    for shape in shapes:
        size = tf.reduce_prod(shape)
        results.append(tf.reshape(flat[offset : offset + size], shape))
        offset += size
    return results


def pcgrad(task_grads):
    """Apply PCGrad projection to a list of per-task gradient lists.

    Parameters
    ----------
    task_grads : list[list[tf.Tensor]]
        ``task_grads[i]`` is a list of gradient tensors (one per trainable
        variable) for loss term *i*.

    Returns
    -------
    list[tf.Tensor]
        The PCGrad-projected gradient (summed over tasks), with the same
        structure as a single element of *task_grads*.
    """
    num_tasks = len(task_grads)
    if num_tasks <= 1:
        return task_grads[0] if task_grads else []

    shapes = [g.shape for g in task_grads[0]]
    flat_grads = tf.stack([_flatten_grads(g) for g in task_grads])  # [T, P]
    norms_sq = tf.reduce_sum(flat_grads ** 2, axis=1)  # [T]

    # Shuffle task processing order. tf.gather reorders rows so that
    # shuffled[i] is the gradient for the i-th task in random order.
    perm = tf.random.shuffle(tf.range(num_tasks))
    shuffled = tf.gather(flat_grads, perm)  # [T, P]
    shuffled_norms_sq = tf.gather(norms_sq, perm)  # [T]

    projected = []
    for i in range(num_tasks):
        gi = shuffled[i]
        for j in range(num_tasks):
            if i == j:
                continue
            dot = tf.reduce_sum(gi * shuffled[j])
            # tf.minimum clamps positive dots to 0, keeping only conflicts.
            coeff = tf.minimum(dot, 0.0) / (shuffled_norms_sq[j] + 1e-12)
            gi = gi - coeff * shuffled[j]
        projected.append(gi)

    total = tf.add_n(projected)
    return _unflatten_grads(total, shapes)


def mgda(task_grads, num_iters=25, eps=1e-8):
    """
    Compute a weighted gradient via MGDA using the Frank-Wolfe algorithm.

    Finds the minimum-norm point in the convex hull of per-task gradients
    iteratively, then returns the corresponding weighted gradient. The
    negative of this point is a descent direction common to all tasks, or
    zero if the current point is Pareto-stationary.

    Parameters
    ----------
    task_grads : list[list[tf.Tensor]]
        ``task_grads[i]`` is a list of gradient tensors (one per trainable
        variable) for loss term *i*. No entry may be None.
    num_iters : int, optional
        Number of Frank-Wolfe iterations. More iterations yield a more
        accurate solution at the cost of compute. Default is 25.
    eps : float, optional
        Small constant added to denominators for numerical stability.
        Default is 1e-8.

    Returns
    -------
    combined_grads : list[tf.Tensor]
        Weighted sum of per-task gradients, one tensor per trainable
        variable. Shapes match those of ``task_grads[i]``.

    Notes
    -----
    The Frank-Wolfe update at each iteration shifts weight toward the
    task whose gradient aligns least with the current mixture, using a
    closed-form line search for the step size gamma in [0, 1].

    Robust near the Pareto front where gradients may become nearly
    linearly dependent, unlike MGDA-II.

    Compatible with ``tf.function``. The Python loop over ``num_iters``
    is unrolled at trace time, so large values of ``num_iters`` will
    increase the size of the traced graph.
    """
    # Plain assert is correct here: len() returns a Python int that is
    # always known at trace time. tf.debugging would be appropriate if
    # we were checking the runtime value of a tensor.
    assert len(task_grads) >= 2, 'Expected at least 2 tasks.'
    n_params = len(task_grads[0])
    assert all(len(g) == n_params for g in task_grads), (
        'All tasks must have the same number of gradient tensors.'
    )

    # Use len() rather than G.shape[0], which may be None if the leading
    # dimension is not statically known by TensorFlow.
    T = len(task_grads)

    # Infer dtype from the gradients so the weight arithmetic is compatible
    # with float16, float32, and float64 models.
    dtype = task_grads[0][0].dtype

    # Flatten each task's gradient list into a single vector so that all
    # parameters are treated as one point in R^P. Stack into [T, P].
    flat = [
        tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        for grads in task_grads
    ]
    G = tf.stack(flat, axis=0)  # [T, P]
    GT = tf.transpose(G)  # [P, T] — hoisted out of the loop

    # Initialise weights uniformly over the T tasks — the starting point
    # in the convex hull is the average of all gradient vectors.
    weights = tf.fill([T], tf.cast(1.0 / T, dtype))

    for _ in range(num_iters):
        # Current mixture gradient: the weighted sum of all task gradients.
        mixture = tf.linalg.matvec(GT, weights)  # [P]

        # Dot product of each task gradient with the mixture. The task with
        # the smallest value conflicts most with the current direction and
        # should receive more weight.
        dots = tf.linalg.matvec(G, mixture)  # [T]

        # Frank-Wolfe linear subproblem: move the full weight to the task
        # whose gradient is most aligned with reducing the mixture norm.
        best = tf.argmin(dots)
        e_t = tf.one_hot(best, T, dtype=dtype)
        diff = e_t - weights  # step direction in weight space

        # Closed-form line search: find the optimal step size gamma in [0,1]
        # by minimising ‖mixture + gamma * G^T diff‖² over gamma.
        d = tf.linalg.matvec(GT, diff)  # [P]
        num = -tf.tensordot(mixture, d, axes=1)
        denom = tf.tensordot(d, d, axes=1) + tf.cast(eps, dtype)
        gamma = tf.clip_by_value(num / denom, 0.0, 1.0)

        # Interpolate weights toward the Frank-Wolfe vertex by gamma.
        # Reassigned as a plain tensor rather than a tf.Variable so this
        # function can be traced by tf.function without error.
        weights = weights + gamma * diff

    # Detach weights from the graph — they are used only to scale gradients,
    # not to propagate second-order information.
    weights = tf.stop_gradient(weights)

    # Reconstruct the weighted gradient in the original parameter shapes by
    # taking the weights-weighted sum across tasks for each variable.
    combined_grads = [
        tf.add_n([weights[i] * task_grads[i][p] for i in range(T)])
        for p in range(len(task_grads[0]))
    ]
    return combined_grads


def mgda_ii(task_grads, scaling_factors=None, eps=1e-8):
    """
    Compute a weighted gradient via MGDA-II using Gram-Schmidt
    orthogonalization.

    Directly constructs the minimum-norm descent direction by applying a
    specially calibrated Gram-Schmidt process to the per-task gradients
    (Désidéri, 2012). The orthogonality of the resulting basis vectors
    allows the convex combination weights to be computed in closed form,
    requiring no iterative solver.

    Parameters
    ----------
    task_grads : list[list[tf.Tensor]]
        ``task_grads[i]`` is a list of gradient tensors (one per trainable
        variable) for loss term *i*. No entry may be None.
    scaling_factors : list[float | tf.Tensor], optional
        Positive scaling constants {S_i}, one per task, used to calibrate
        the Gram-Schmidt normalization. Controls the relative influence of
        each task's gradient on the descent direction. Defaults to the L2
        norm of each task's flattened gradient vector.
    eps : float, optional
        Small constant added to denominators for numerical stability, and
        used as a fallback scaling when the Gram-Schmidt denominator A_i
        is near zero. Default is 1e-8.

    Returns
    -------
    combined_grads : list[tf.Tensor]
        Weighted sum of per-task gradients, one tensor per trainable
        variable. Shapes match those of ``task_grads[i]``.

    Notes
    -----
    The Gram-Schmidt calibration follows equations (7)-(10) of Désidéri
    (2012). The orthogonal basis {u_i} satisfies:

        u_1 = J'_1 / S_1
        u_i = (J'_i - sum_{k<i} c_{i,k} u_k) / A_i

    where c_{i,k} = <J'_i, u_k> / <u_k, u_k> and
    A_i = S_i - sum_{k<i} c_{i,k}.

    The closed-form weights over this basis are then:

        alpha_i = (1 / ||u_i||^2) / sum_j (1 / ||u_j||^2)

    Uses modified Gram-Schmidt (projecting the running residual rather than
    the original gradient at each step) for improved numerical stability
    over the classical formulation in the paper.

    Requires the task gradients to be linearly independent. May become
    numerically unstable near the Pareto front as gradients approach
    linear dependence; prefer `mgda_frank_wolfe` in that regime.

    References
    ----------
    Désidéri, J.-A. (2012). MGDA II: A direct method for calculating a
    descent direction common to several criteria. INRIA Research Report
    RR-7922. https://hal.science/hal-00685762
    """
    # Plain assert is correct here: len() returns a Python int that is
    # always known at trace time. tf.debugging would be appropriate if
    # we were checking the runtime value of a tensor.
    assert len(task_grads) >= 2, 'Expected at least 2 tasks.'
    n_params = len(task_grads[0])
    assert all(len(g) == n_params for g in task_grads), (
        'All tasks must have the same number of gradient tensors.'
    )
    if scaling_factors is not None:
        assert len(scaling_factors) == len(task_grads), (
            'scaling_factors must have one entry per task.'
        )

    n = len(task_grads)

    # Infer dtype from the gradients so the Gram-Schmidt arithmetic is
    # compatible with float16, float32, and float64 models.
    dtype = task_grads[0][0].dtype
    eps_t = tf.cast(eps, dtype)

    # Flatten each task's gradient list into a single vector so that the
    # Gram-Schmidt process operates in the full parameter space R^P.
    flat = [
        tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        for grads in task_grads
    ]

    # Default scaling: use each gradient's L2 norm so that all tasks start
    # on equal footing regardless of their gradient magnitudes.
    if scaling_factors is None:
        scaling_factors = [tf.norm(g) + eps_t for g in flat]
    S = [tf.cast(s, dtype) for s in scaling_factors]

    # Modified Gram-Schmidt with Désidéri's calibration (equations 7-10).
    # Produces orthogonal vectors {u_i} spanning the same subspace as the
    # original gradients, with a normalisation chosen so that the min-norm
    # point of their convex hull can be written in closed form.
    #
    # Modified (rather than classical) Gram-Schmidt projects the running
    # residual onto each u_k rather than the original flat[i]. Both are
    # mathematically equivalent but modified Gram-Schmidt is more numerically
    # stable in finite precision.
    #
    # u_k norms squared are cached as they are produced to avoid recomputing
    # them for each subsequent task in the inner loop.
    us = []
    u_norms_sq = []
    for i in range(n):
        residual = flat[i]
        coeff_sum = tf.zeros(
            (), dtype=dtype
        )  # scalar zero in the correct dtype

        # Subtract projections onto all previously computed u_k using the
        # running residual (modified Gram-Schmidt). coeff_sum accumulates
        # the sum of projection coefficients for the denominator A_i.
        for k in range(i):
            # Use cached ‖u_k‖² rather than recomputing it each iteration.
            c_ik = tf.tensordot(residual, us[k], axes=1) / (
                u_norms_sq[k] + eps_t
            )
            residual = residual - c_ik * us[k]
            coeff_sum = coeff_sum + c_ik

        # A_i = S_i - sum_{k<i} c_{i,k} is the Désidéri calibration factor.
        # If it collapses to zero the gradient is nearly linearly dependent on
        # the previous ones; fall back to eps * S_i to avoid division by zero.
        # S_i is captured in a local variable to avoid a closure bug when this
        # code is traced by tf.function — lambdas closed over loop variables
        # would otherwise all reference the final value of i.
        S_i = S[i]
        A_i = tf.cond(
            tf.abs(S_i - coeff_sum) > eps_t,
            lambda: S_i - coeff_sum,
            lambda: eps_t * S_i,
        )
        u_i = residual / A_i
        us.append(u_i)
        u_norms_sq.append(tf.tensordot(u_i, u_i, axes=1))

    # Closed-form weights: because the u_i are orthogonal, minimising
    # ‖Σ α_i u_i‖² subject to Σ α_i = 1 gives α_i ∝ 1/‖u_i‖².
    # Tasks whose orthogonalised vector is large (high residual energy)
    # receive less weight in the final direction.
    inv_sq = [1.0 / (norm_sq + eps_t) for norm_sq in u_norms_sq]
    total = tf.add_n(inv_sq)
    alphas = tf.stop_gradient(tf.stack([v / total for v in inv_sq]))  # [n]

    # Reconstruct the weighted gradient in the original parameter shapes by
    # taking the alpha-weighted sum across tasks for each variable.
    combined_grads = [
        tf.add_n([alphas[i] * task_grads[i][p] for i in range(n)])
        for p in range(len(task_grads[0]))
    ]
    return combined_grads
