from typing import TypeVar, Union

import numpy as np

from pyscf.afqmc import config

use_jax = config.afqmc_config["use_jax"]

if use_jax:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    import jax.scipy as jsp
    from jax import (
        Array,
        checkpoint,
        custom_jvp,
        dtypes,
        grad,
        jit,
        jvp,
        lax,
        vjp,
        vmap,
    )
else:
    import scipy as sp

    jsp = sp  # Use scipy as jsp
    Array = np.ndarray
    from numpy import dtypes

    class DummyJax:
        Array = np.ndarray

        def __getattr__(self, name):
            raise AttributeError(
                f"JAX is not installed. The feature 'jax.{name}' is not available."
            )

    jax = DummyJax()

    class ArrayAtIndexer:
        def __init__(self, array):
            self.array = array

        def set(self, value):
            self.array[self.indices] = value
            return self.array.copy()

        def __getitem__(self, indices):
            self.indices = indices
            return self

    class DummyArrayType(np.ndarray):
        @property
        def at(self):
            return ArrayAtIndexer(self)

    # Proxy for nested modules like linalg
    class SubmoduleProxy:
        def __init__(self, np_submodule):
            self.np_submodule = np_submodule

        def __getattr__(self, name):
            np_func = getattr(self.np_submodule, name)

            def wrapped(*args, **kwargs):
                result = np_func(*args, **kwargs)
                if isinstance(result, (np.ndarray, tuple)):
                    if isinstance(result, tuple):
                        return tuple(
                            r.view(DummyArrayType) if isinstance(r, np.ndarray) else r
                            for r in result
                        )
                    return result.view(DummyArrayType)
                return result

            return wrapped

    class JaxNumopyProxy:
        def __init__(self):
            self.np = np
            # Pre-create common submodules
            self.linalg = SubmoduleProxy(np.linalg)
            self.fft = SubmoduleProxy(np.fft)

        def array(self, *args, **kwargs):
            return np.array(*args, **kwargs).view(DummyArrayType)

        def __getattr__(self, name):
            np_attr = getattr(np, name)
            if callable(np_attr):

                def wrapped(*args, **kwargs):
                    result = np_attr(*args, **kwargs)
                    if isinstance(result, np.ndarray):
                        return result.view(DummyArrayType)
                    return result

                return wrapped
            return np_attr

    jnp = JaxNumopyProxy()

    def jit(fun, static_argnums=None, static_argnames=None, **kwargs):
        # Ignore static_argnums and other JAX-specific arguments
        return fun

    def checkpoint(fun, *, prevent_cse=True, static_argnums=(), policy=None):
        """Dummy checkpoint that just returns the function unchanged."""
        return fun

    # Create a dummy attribute for defjvp
    class DummyJVPDecorator:
        def __call__(self, *args, **kwargs):
            def decorator(jvp_rule):
                return jvp_rule

            return decorator

    class CustomJvpFunction:
        def __init__(self, fun):
            self.fun = fun
            # Attach dummy defjvp
            self.defjvp = DummyJVPDecorator()

        def __call__(self, *args, **kwargs):
            return self.fun(*args, **kwargs)

    def custom_jvp(fun=None, *, nondiff_argnums=()):
        if fun is None:
            return lambda f: CustomJvpFunction(f)
        return CustomJvpFunction(fun)

    def grad(f):
        raise NotImplementedError("grad requires JAX - install JAX to use this feature")

    def vjp(fun, *primals):
        raise NotImplementedError(
            "Vector-Jacobian product (vjp) requires JAX - install JAX to use automatic differentiation"
        )

    def jvp(fun, primals, tangents):
        raise NotImplementedError(
            "Jacobian-vector product (jvp) requires JAX - install JAX to use automatic differentiation"
        )

    def vmap(f, in_axes=0, out_axes=0):
        def wrapped(*args):
            # Handle in_axes
            if isinstance(in_axes, int):
                axes = [in_axes] * len(args)
            else:
                axes = in_axes

            # Get the batch size from the first mapped argument
            for arg, axis in zip(args, axes):
                if axis is not None:
                    batch_size = arg.shape[axis]
                    break

            # Prepare sliced arguments for each batch element
            batched_args = []
            for arg, axis in zip(args, axes):
                if axis is None:
                    batched_args.append([arg] * batch_size)
                else:
                    slices = []
                    for i in range(batch_size):
                        idx = [slice(None)] * arg.ndim
                        idx[axis] = i
                        slices.append(arg[tuple(idx)])
                    batched_args.append(slices)

            # Apply function to each slice
            results = []
            for batch_idx in range(batch_size):
                batch_args = [args[batch_idx] for args in batched_args]
                results.append(f(*batch_args))

            # Store out_axes locally
            out_ax = out_axes  # Make local copy to avoid UnboundLocalError

            # Handle out_axes for tuple returns
            if isinstance(results[0], tuple):
                # Split results into separate lists for each tuple element
                split_results = list(zip(*results))
                if isinstance(out_ax, int):
                    # Same axis for all outputs
                    return tuple(np.stack(r, axis=out_ax) for r in split_results)
                else:
                    # Different axis for each output
                    return tuple(
                        np.stack(r, axis=ax) for r, ax in zip(split_results, out_ax)
                    )
            else:
                # Single output
                if isinstance(out_ax, tuple):
                    out_ax = out_ax[0]
                return np.stack(results, axis=out_ax)

        return wrapped

    class LaxModule:
        def scan(self, f, init, xs, length=None, reverse=False, unroll=1):
            if length is not None:
                xs_iter = range(length)
            else:
                if isinstance(xs, tuple):
                    xs_iter = range(len(xs[0]))
                else:
                    xs_iter = range(len(xs))

            if reverse:
                xs_iter = reversed(list(xs_iter))

            carry = init
            ys = []

            for i in xs_iter:
                if length is not None:
                    result = f(carry, xs)
                else:
                    if isinstance(xs, tuple):
                        current_xs = tuple(x[i] for x in xs)
                        result = f(carry, current_xs)
                    else:
                        result = f(carry, xs[i])

                # Handle tuple outputs
                if isinstance(result, tuple):
                    carry = result[0]
                    y = result[1]  # This could itself be a tuple
                else:
                    carry, y = (
                        result  # If not tuple, assume it returns exactly 2 values
                    )

                ys.append(y)

            if reverse:
                ys = ys[::-1]

            # Handle stacking of possibly tuple outputs
            if ys and isinstance(ys[0], tuple):
                # Split tuple outputs into separate lists
                split_ys = list(zip(*ys))
                # Stack each component separately
                stacked_ys = tuple(np.array(component) for component in split_ys)
                return carry, stacked_ys
            else:
                return carry, np.array(ys)

    lax = LaxModule()

    import numpy.random as np_random

    class PRNGKey:
        def __init__(self, seed):
            self.rng = np_random.RandomState(seed)

    class RandomModule:
        def PRNGKey(self, seed):
            return PRNGKey(seed)

        def split(self, key, num=2):
            seeds = key.rng.randint(0, 2**31, size=num)
            return [PRNGKey(seed) for seed in seeds]

        def uniform(self, key, shape=(), dtype=None, minval=0.0, maxval=1.0):
            return key.rng.uniform(minval, maxval, size=shape)

        def normal(self, key, shape=(), dtype=None):
            return key.rng.normal(0, 1, size=shape)

        def randint(self, key, shape, minval, maxval):
            return key.rng.randint(minval, maxval, size=shape)

        # Add other commonly used distributions
        def exponential(self, key, shape=(), dtype=None):
            return key.rng.exponential(size=shape)

        def bernoulli(self, key, p=0.5, shape=()):
            return key.rng.binomial(1, p, size=shape)

    random = RandomModule()


ArrayLike = Union[Array, np.ndarray]
