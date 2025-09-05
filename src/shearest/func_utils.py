import jax.numpy as jnp

def clip_by_l2_norm(x):
    norm = jnp.linalg.norm(x, axis=0)
    scale = jnp.minimum(1.0, 1. / (norm + 1e-2))  # adding epsilon for numerical stability and enforcing norm < 1.
    return x * scale

def to_unit_disk(x, M=0.99):
    z1, z2 = x
    # sample z1,z2 ~ Normal(0,1)
    r = jnp.linalg.norm(x, axis=0)
    # squash radius smoothly into (0, M)
    r_squash = M * jnp.tanh(r)
    factor = r_squash / (r + 1e-12)
    return x * factor

def complex_2_stack(x):
    """Convert a complex image to a stack of real and imaginary parts."""
    return jnp.stack([jnp.real(x), jnp.imag(x)], axis=0)

def stack_2_complex(x, batch=None):
    """Convert a stack of real and imaginary parts to a complex image."""
    if batch is None:
        return jnp.complex64(x[0] + 1j * x[1])
    else:
        return jnp.complex64(x[:,0] + 1j * x[:,1])