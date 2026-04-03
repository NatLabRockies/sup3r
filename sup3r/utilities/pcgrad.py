"""PCGrad (Projected Conflicting Gradients) for TensorFlow.

Implements the gradient surgery algorithm from:
    Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

When training with multiple loss terms, standard practice sums per-task
gradients.  If two task gradients conflict (negative dot product), this can
slow or destabilize training.  PCGrad projects each task gradient onto the
normal plane of every other conflicting task gradient before summing,
eliminating the destructive interference.
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
        tf.zeros_like(r) if g is None else g
        for g, r in zip(grads, ref_grads)
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
        results.append(tf.reshape(flat[offset:offset + size], shape))
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

    # Random task ordering (re-sampled every call) to avoid bias.
    indices = tf.random.shuffle(tf.range(num_tasks))

    projected = []
    for idx in range(num_tasks):
        i = indices[idx]
        gi = tf.gather(tf.stack(flat_grads), i)
        for jdx in range(num_tasks):
            j = indices[jdx]
            gj = tf.gather(tf.stack(flat_grads), j)
            same = tf.equal(i, j)
            dot = tf.reduce_sum(gi * gj)
            proj = gi - (dot / (tf.reduce_sum(gj * gj) + 1e-12)) * gj
            # Project onto normal plane of gj when gradients conflict.
            gi = tf.cond(
                tf.logical_or(same, dot >= 0),
                true_fn=lambda gi=gi: gi,
                false_fn=lambda proj=proj: proj,
            )
        projected.append(gi)

    total = tf.add_n(projected)
    return _unflatten_grads(total, shapes)
