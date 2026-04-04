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
    flat_grads = [_flatten_grads(g) for g in task_grads]

    stacked = tf.stack(flat_grads)  # (num_tasks, D)
    norms_sq = tf.reduce_sum(stacked**2, axis=1)  # (num_tasks,)

    projected = []
    order = tf.random.shuffle(tf.range(num_tasks)).numpy()
    for i in order:
        gi = flat_grads[i]
        dots = tf.reduce_sum(stacked * gi[tf.newaxis, :], axis=1)
        # Mask: conflicting tasks (dot < 0) excluding self.
        conflict = dots < 0
        conflict = tf.tensor_scatter_nd_update(conflict, [[i]], [False])
        if tf.reduce_any(conflict):
            coeffs = dots / (norms_sq + 1e-12)
            # Zero out non-conflicting entries.
            coeffs = tf.where(conflict, coeffs, tf.zeros_like(coeffs))
            gi = gi - tf.reduce_sum(coeffs[:, tf.newaxis] * stacked, axis=0)  # noqa: PLR6104
        projected.append(gi)

    total = tf.add_n(projected)
    return _unflatten_grads(total, shapes)


def mgda(task_grads):
    r"""MGDA-II: direct min-norm solver for multi-task gradients.

    Finds the minimum-norm point in the convex hull of per-task
    gradients using an active-set QP solver on the :math:`T \times T`
    Gram matrix (Désidéri, 2012).  At each step the equality-constrained
    sub-problem is solved analytically via :math:`\alpha = G^{-1}\mathbf{1} /
    (\mathbf{1}^T G^{-1}\mathbf{1})` and tasks with negative weights are
    removed until all weights are non-negative.  Converges in at most *T*
    iterations.

    Parameters
    ----------
    task_grads : list[list[tf.Tensor]]
        ``task_grads[i]`` is a list of gradient tensors (one per trainable
        variable) for loss term *i*.  Same format as :func:`pcgrad`.

    Returns
    -------
    list[tf.Tensor]
        The MGDA-II-combined gradient, with the same structure as a single
        element of *task_grads*.
    """
    num_tasks = len(task_grads)
    if num_tasks <= 1:
        return task_grads[0] if task_grads else []

    shapes = [g.shape for g in task_grads[0]]
    flat_grads = [_flatten_grads(g) for g in task_grads]
    stacked = tf.stack(flat_grads)  # (num_tasks, D)

    # Gram matrix:  G[i,j] = <g_i, g_j>
    gram = tf.matmul(stacked, stacked, transpose_b=True)  # (T, T)

    # Active-set QP:  min  alpha^T G alpha
    #                  s.t. alpha >= 0,  sum(alpha) = 1
    # Analytical solution for the equality-only sub-problem:
    #   alpha = G_A^{-1} 1  /  (1^T G_A^{-1} 1)
    active = list(range(num_tasks))

    for _ in range(num_tasks):
        n = len(active)
        if n == 1:
            break
        idx = tf.constant(active)
        g_sub = tf.gather(tf.gather(gram, idx), idx, axis=1)
        # Small diagonal regularisation for numerical stability.
        g_sub = g_sub + 1e-8 * tf.eye(n, dtype=g_sub.dtype)

        ones = tf.ones([n, 1], dtype=g_sub.dtype)
        sol = tf.linalg.solve(g_sub, ones)  # G^{-1} 1
        alpha_sub = tf.squeeze(sol / tf.reduce_sum(sol), axis=1)

        if tf.reduce_min(alpha_sub) >= -1e-8:
            # All weights non-negative → feasible.
            alpha_sub = tf.maximum(alpha_sub, 0.0)
            alpha_sub = alpha_sub / tf.reduce_sum(alpha_sub)
            alpha = tf.scatter_nd(
                tf.expand_dims(idx, 1),
                alpha_sub,
                [num_tasks],
            )
            break
        else:
            # Drop the most negative weight and re-solve.
            drop = tf.argmin(alpha_sub).numpy()
            active.pop(drop)
    else:
        # Single task remaining after loop exhaustion.
        alpha = tf.one_hot(active[0], num_tasks, dtype=stacked.dtype)

    total = tf.linalg.matvec(stacked, alpha, transpose_a=True)  # (D,)
    return _unflatten_grads(total, shapes)
