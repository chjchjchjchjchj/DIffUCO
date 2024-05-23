



if(__name__ == "__main__"):
    import numpy as jnp
    from matplotlib import pyplot as plt

    N = 100
    dim = 15
    L = 1.41
    edges = jnp.linspace(0, L, N)[:, None]

    i = jnp.arange(0, dim, 2)
    L_prime_sin = L * ((dim - i + 1) / dim)[None, :]
    L_prime_cos = L * ((dim - i) / dim)[None, :]
    cos_edges = jnp.cos(2 * jnp.pi * edges / L_prime_cos)
    sin_edges = jnp.sin(2 * jnp.pi * edges / L_prime_sin)

    for j, ii in enumerate(i):
        plt.figure()
        plt.title(str(j))
        plt.plot(jnp.arange(0, N), cos_edges[:, j])
        plt.plot(jnp.arange(0, N), sin_edges[:, j])
        plt.show()