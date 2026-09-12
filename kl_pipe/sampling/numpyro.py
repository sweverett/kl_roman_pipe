"""
NumPyro gradient-based MCMC backend.

NumPyro provides robust JAX-native NUTS/HMC with superior mass matrix
adaptation compared to raw BlackJAX. This is the recommended backend for
joint velocity+intensity models where parameter gradients span multiple
orders of magnitude.

Key Features
------------
- **Z-score reparameterization**: Automatic scaling to O(1) latent space
- **Dense mass matrix**: Handles parameter correlations
- **R-hat and ESS diagnostics**: Built-in convergence assessment
- **Multi-chain support**: Sequential, parallel, or vectorized execution

Architecture
------------
The sampler wraps the existing JAX log-posterior from InferenceTask using
``numpyro.factor()``. The Z-score reparameterization samples in a standardized
latent space (all parameters ~N(0,1)), then transforms to physical space
before evaluating the likelihood.

.. note::
   This sampler uses ``task.get_log_posterior_fn()`` which INCLUDES priors.
   The latent Normal(0,1) variables are purely for numerical conditioning,
   not for specifying priors. Do not add informative numpyro.sample() calls
   as this would double-count the prior.

References
----------
NumPyro documentation: https://num.pyro.ai/
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Dict, Optional, Callable, Tuple, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import NumpyroSamplerConfig, ReparamStrategy
from kl_pipe.priors import (
    Prior,
    CircularUniform,
    ConditionalLogNormal,
    Gaussian,
    TruncatedNormal,
    TruncatedNormalMixture,
    TruncatedLogNormal,
    Uniform,
    LogUniform,
    LogNormal,
)

if TYPE_CHECKING:
    from kl_pipe.sampling.task import InferenceTask, LaplacePreconditioner


def compute_reparam_scales(
    prior: Prior, name: str, parent_loc: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute (loc, scale) for Z-score reparameterization.

    Maps each prior type to appropriate centering and scaling values
    so that sampling in z ~ N(0,1) explores the prior support efficiently.

    Parameters
    ----------
    prior : Prior
        The prior distribution for this parameter.
    name : str
        Parameter name (for error messages).

    Returns
    -------
    loc : float
        Center point in physical space.
    scale : float
        Characteristic scale in physical space.

    Notes
    -----
    For bounded priors (Uniform, TruncatedNormal), the bounds are still
    enforced by the prior log_prob returning -inf outside support.
    The scaling just ensures the sampler starts in the right ballpark.
    """
    if isinstance(prior, Gaussian):
        # Standard case: use prior parameters directly
        return float(prior.mu), float(prior.sigma)

    elif isinstance(prior, TruncatedNormal):
        # Use the UNDERLYING Gaussian parameters
        # The truncation is enforced by prior.log_prob() returning -inf
        return float(prior.mu), float(prior.sigma)

    elif isinstance(prior, Uniform):
        # Center at midpoint, scale so ±2σ approximately covers the range
        loc = (prior.low + prior.high) / 2
        scale = (prior.high - prior.low) / 4  # 4σ spans the range
        return float(loc), float(scale)

    elif isinstance(prior, LogNormal):
        # center at the distribution mean, scale = its standard deviation
        return prior.mean, prior.std

    elif isinstance(prior, LogUniform):
        # Work in log-space conceptually
        # Geometric mean as center
        log_mid = (jnp.log(prior.low) + jnp.log(prior.high)) / 2
        log_scale = (jnp.log(prior.high) - jnp.log(prior.low)) / 4
        return float(jnp.exp(log_mid)), float(jnp.exp(log_mid) * log_scale)

    elif isinstance(prior, TruncatedLogNormal):
        # underlying log-normal moments; truncation is enforced by log_prob
        median = float(math.exp(prior.mu))
        return median, median * float(prior.sigma)

    elif isinstance(prior, TruncatedNormalMixture):
        # weighted mean and total mixture spread; this is a conditioning
        # hint, so a single centre for a bimodal prior is adequate
        rec = prior.to_dict()
        return float(rec['loc']), float(rec['scale'])

    elif isinstance(prior, ConditionalLogNormal):
        # centre on the parent's own centre scaled by the median ratio; the
        # parent is sampled, so this is a static stand-in for conditioning
        # only, not part of the density
        if parent_loc is None:
            raise ValueError(
                f"conditional prior '{name}' needs its parent's centre to "
                f"compute a reparameterization scale; pass parent_loc"
            )
        loc = float(prior.ratio_median * parent_loc)
        return loc, loc * float(prior.sigma_ratio)

    elif isinstance(prior, CircularUniform):
        # centre at the half-period; 4 sigma spans one period
        return float(prior.period / 2), float(prior.period / 4)

    else:
        raise TypeError(f"Unknown prior type for '{name}': {type(prior)}")


# Chunk size for the end-of-sampling log-posterior evaluation. Every
# intermediate of the vmapped likelihood (render grids, dispersal tensors)
# carries the chunk as a batch dimension, ~12-20 MB per sample on the
# production fit, so the transient peak is chunk_size x that: 16 keeps it
# under ~0.3 GiB, which packed GPU workers can always satisfy.
_LOG_PROB_CHUNK_SIZE = 16


def _batched_log_posterior_chunked(
    log_posterior_jittable: Callable,
    samples: np.ndarray,
    chunk_size: int = _LOG_PROB_CHUNK_SIZE,
) -> np.ndarray:
    """Evaluate the log-posterior over many samples in fixed-size chunks.

    Equivalent result to ``jax.vmap(fn)(samples)`` but with peak memory bounded
    by ``chunk_size`` rather than ``len(samples)``. See ``_LOG_PROB_CHUNK_SIZE``.

    Parameters
    ----------
    log_posterior_jittable : callable
        Single-sample log-posterior ``theta -> scalar``.
    samples : np.ndarray
        Array of shape ``(n_total, n_params)``.
    chunk_size : int
        Number of samples evaluated per vmap call.

    Returns
    -------
    np.ndarray
        Log-posterior values, shape ``(n_total,)``.
    """
    fn = jax.jit(jax.vmap(log_posterior_jittable))
    n = samples.shape[0]
    out = []
    for start in range(0, n, chunk_size):
        chunk = jnp.asarray(samples[start : start + chunk_size])
        out.append(np.asarray(fn(chunk)))
    return np.concatenate(out)


class NumpyroSampler(Sampler):
    """
    NumPyro gradient-based sampler with Z-score reparameterization.

    This sampler wraps the existing JAX log-posterior using numpyro.factor()
    and applies automatic Z-score reparameterization to normalize parameter
    scales. This is critical for joint models where gradients can vary by
    10^4+ across parameters.

    Advantages over BlackJAX
    ------------------------
    - More robust mass matrix adaptation
    - Built-in R-hat and ESS diagnostics
    - Better handling of challenging posteriors
    - Z-score reparameterization for multi-scale problems

    Parameters
    ----------
    task : InferenceTask
        The inference task to solve.
    config : NumpyroSamplerConfig
        Sampler configuration options.

    Attributes
    ----------
    requires_gradients : bool
        True - NumPyro NUTS/HMC uses gradients.
    provides_evidence : bool
        False - NumPyro MCMC does not compute evidence.
    config_class : type
        NumpyroSamplerConfig

    Examples
    --------
    >>> from kl_pipe.sampling import InferenceTask, NumpyroSamplerConfig
    >>> from kl_pipe.sampling.numpyro import NumpyroSampler
    >>>
    >>> config = NumpyroSamplerConfig(
    ...     n_samples=2000,
    ...     n_warmup=1000,
    ...     n_chains=4,
    ...     seed=42,
    ... )
    >>> sampler = NumpyroSampler(task, config)
    >>> result = sampler.run()

    Laplace preconditioning (opt-in) -- ~2x faster warmup on correlated joint
    posteriors by initializing NUTS at the MAP with a fixed inverse-Hessian
    mass matrix:

    >>> # config flag: sampler computes the MAP + Hessian internally
    >>> config = NumpyroSamplerConfig(precondition='laplace', n_warmup=100)
    >>> result = build_sampler('numpyro', task, config).run()
    >>>
    >>> # composable: compute the preconditioner once, reuse / inspect it
    >>> pre = task.laplace_preconditioner()
    >>> sampler = NumpyroSampler(task, config, preconditioner=pre)
    >>> result = sampler.run()

    See Also
    --------
    BlackJAXSampler : Simpler but less robust gradient-based sampler.
    """

    requires_gradients = True
    provides_evidence = False
    config_class = NumpyroSamplerConfig

    def __init__(
        self,
        task: 'InferenceTask',
        config: NumpyroSamplerConfig,
        preconditioner: Optional['LaplacePreconditioner'] = None,
    ):
        super().__init__(task, config)
        self._reparam_scales: Optional[Dict[str, Tuple[float, float]]] = None
        # Optional precomputed Laplace preconditioner. If config.precondition
        # == 'laplace' and none is supplied, run() computes one.
        self._preconditioner = preconditioner

    def _compute_reparam_scales(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute reparameterization scales based on config strategy.

        Returns
        -------
        dict
            Mapping from parameter name to (loc, scale) tuple.
        """
        strategy = self.config.reparam_strategy

        if strategy == ReparamStrategy.NONE:
            # No reparameterization: identity transform
            return {name: (0.0, 1.0) for name in self.task.sampled_names}

        elif strategy == ReparamStrategy.PRIOR:
            # Use prior statistics. Topological order so a conditional prior's
            # parent already has a centre when the child is reached.
            priors = self.task.priors
            parents = priors.conditional_parents
            scales = {}
            for name in priors.topological_order:
                prior = priors.get_prior(name)
                parent = parents.get(name)
                scales[name] = compute_reparam_scales(
                    prior,
                    name,
                    parent_loc=None if parent is None else scales[parent][0],
                )
            return scales

        else:
            raise ValueError(f"Unknown reparam strategy: {strategy}")

    def _build_numpyro_model(
        self,
        reparam_scales: Dict[str, Tuple[float, float]],
    ) -> Callable:
        """
        Build NumPyro model that wraps task's log_posterior.

        IMPORTANT: Uses task.get_log_posterior_fn() which INCLUDES the prior.
        We sample from uninformative Normal(0,1) in latent space purely for
        numerical conditioning - the actual prior is in the task's log_prob.

        Parameters
        ----------
        reparam_scales : dict
            Mapping from parameter name to (loc, scale) for Z-score transform.

        Returns
        -------
        callable
            NumPyro model function.
        """
        import numpyro
        import numpyro.distributions as dist

        task = self.task
        log_posterior_fn = task.get_log_posterior_fn()
        sampled_names = task.sampled_names

        def model():
            theta_physical = []

            for name in sampled_names:
                loc, scale = reparam_scales[name]

                # Sample in latent space - this is NOT the prior!
                # Using improper flat "prior" in z-space; actual prior in log_posterior
                z = numpyro.sample(f"_z_{name}", dist.Normal(0.0, 1.0))

                # Transform to physical space
                param = loc + scale * z

                # Store as deterministic for output
                numpyro.deterministic(name, param)
                theta_physical.append(param)

            # Stack into theta array (in sampled_names order, which matches task)
            theta = jnp.stack(theta_physical)

            # Evaluate log posterior (includes both likelihood AND prior)
            log_post = log_posterior_fn(theta)

            # Add to model via factor
            numpyro.factor("log_posterior", log_post)

        return model

    def _get_init_params(
        self,
        rng_key: jax.Array,
        reparam_scales: Dict[str, Tuple[float, float]],
    ) -> Optional[Dict[str, jnp.ndarray]]:
        """
        Get initial parameters for MCMC chains.

        Parameters
        ----------
        rng_key : jax.Array
            Random key.
        reparam_scales : dict
            Mapping from parameter name to (loc, scale).

        Returns
        -------
        dict or None
            Initial values for latent z parameters. Returns None to let
            NumPyro initialize from the prior (which is Normal(0,1) for z).
        """
        init_strategy = self.config.init_strategy
        sampled_names = self.task.sampled_names

        # For 'prior' strategy with our setup, z~N(0,1) IS the prior in latent space
        # So we can just let NumPyro initialize from its prior
        if init_strategy == 'prior':
            # Let NumPyro handle initialization from the model's prior
            # Since we sample z ~ Normal(0, 1), this is equivalent to prior init
            return None

        elif init_strategy == 'median':
            # Start at z=0 (which maps to prior means in physical space)
            init_params = {}
            for name in sampled_names:
                init_params[f"_z_{name}"] = jnp.array(0.0)
            return init_params

        elif init_strategy == 'jitter':
            # Small random perturbation around z=0
            keys = random.split(rng_key, len(sampled_names))
            init_params = {}
            for i, name in enumerate(sampled_names):
                z_init = 0.1 * random.normal(keys[i], ())
                init_params[f"_z_{name}"] = z_init
            return init_params

        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

    def _collect_diagnostics(
        self,
        mcmc,
        reparam_scales: Optional[Dict[str, Tuple[float, float]]],
        samples_by_chain: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """
        Collect all diagnostics from NumPyro MCMC run.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC
            Completed MCMC run.
        reparam_scales : dict
            The scales used for reparameterization.

        Returns
        -------
        dict
            Comprehensive diagnostics.
        """
        from numpyro.diagnostics import summary as numpyro_summary

        extra_fields = mcmc.get_extra_fields()

        # Physical-space per-chain samples for R-hat/ESS. The preconditioned
        # (potential_fn) path supplies these explicitly since get_samples()
        # returns a flat array there rather than named sites.
        if samples_by_chain is not None:
            physical_samples = {
                name: np.array(arr) for name, arr in samples_by_chain.items()
            }
        else:
            samples = mcmc.get_samples(group_by_chain=True)
            physical_samples = {}
            for name in self.task.sampled_names:
                if name in samples:
                    physical_samples[name] = np.array(samples[name])

        # Compute R-hat and ESS using numpyro's built-in functions
        summary_dict = numpyro_summary(physical_samples)

        r_hat = {}
        ess = {}
        for name in self.task.sampled_names:
            if name in summary_dict:
                r_hat[name] = float(summary_dict[name]['r_hat'])
                ess[name] = float(summary_dict[name]['n_eff'])

        # Divergences
        diverging = np.array(extra_fields.get('diverging', []))
        n_divergences = int(diverging.sum()) if diverging.size > 0 else 0

        # Acceptance probabilities
        accept_prob = extra_fields.get('accept_prob', None)
        if accept_prob is not None:
            accept_prob = np.array(accept_prob)
            mean_accept = float(accept_prob.mean())
        else:
            mean_accept = None

        # Number of leapfrog steps (tree depth proxy)
        num_steps = extra_fields.get('num_steps', None)
        if num_steps is not None:
            num_steps = np.array(num_steps)
            mean_tree_depth = float(np.log2(num_steps + 1).mean())
        else:
            mean_tree_depth = None

        # Step size from last state
        try:
            step_size = float(mcmc.last_state.adapt_state.step_size)
        except (AttributeError, TypeError):
            step_size = None

        diagnostics = {
            # Divergence info
            'diverging': diverging,
            'n_divergences': n_divergences,
            'divergence_rate': (
                n_divergences / diverging.size if diverging.size > 0 else 0.0
            ),
            # Acceptance
            'accept_prob': accept_prob,
            'mean_accept_prob': mean_accept,
            # Tree depth / steps
            'num_steps': num_steps,
            'mean_tree_depth': mean_tree_depth,
            # Adaptation
            'step_size': step_size,
            # Convergence diagnostics
            'r_hat': r_hat,
            'ess': ess,
            # Reparameterization info
            'reparam_strategy': self.config.reparam_strategy.value,
            'reparam_scales': reparam_scales,
        }

        # Optionally save mass matrix
        if self.config.save_mass_matrix:
            try:
                mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix
                diagnostics['inverse_mass_matrix'] = np.array(mass_matrix)
            except (AttributeError, TypeError):
                pass

        return diagnostics

    def _resolve_chain_method(self, n_chains: int) -> str:
        """Resolve ``config.chain_method`` to a concrete numpyro chain method.

        An explicit non-None ``config.chain_method`` always wins (including
        raising if 'parallel' is requested without enough devices). With
        ``config.chain_method=None`` (auto), dispatch by backend and device
        count:

        - non-CPU backend -> 'vectorized'
        - CPU with ``jax.local_device_count() >= n_chains`` and more than one
          device available -> 'parallel'
        - CPU otherwise -> 'sequential'

        Parameters
        ----------
        n_chains : int
            Number of chains this run will launch.

        Returns
        -------
        str
            One of 'sequential', 'parallel', 'vectorized'.

        Raises
        ------
        ValueError
            If ``config.chain_method='parallel'`` is requested explicitly
            but ``jax.local_device_count() < n_chains``. NumPyro itself
            only warns and silently falls back to sequential execution in
            this case; kl_pipe raises instead so a misconfigured device
            count is never silent.
        """
        method = self.config.chain_method
        if method is not None:
            if method == 'parallel' and jax.local_device_count() < n_chains:
                raise ValueError(
                    f"chain_method='parallel' requires "
                    f"jax.local_device_count() >= n_chains ({n_chains}), "
                    f"got {jax.local_device_count()} device(s). Set the "
                    f"KLPIPE_CPU_DEVICES environment variable (see "
                    f"kl_pipe._devices) to force more host CPU devices "
                    f"before the first kl_pipe/JAX import, or use "
                    f"chain_method='sequential'/'vectorized' instead."
                )
            return method

        if jax.default_backend() != 'cpu':
            return 'vectorized'
        n_devices = jax.local_device_count()
        if n_devices >= n_chains and n_devices > 1:
            return 'parallel'
        return 'sequential'

    def run(self) -> SamplerResult:
        """
        Run NumPyro NUTS sampler.

        Returns
        -------
        SamplerResult
            Posterior samples and diagnostics.
        """
        import numpyro
        from numpyro.infer import MCMC, NUTS

        # Laplace-preconditioned path (opt-in). Physical-space potential_fn NUTS
        # with a fixed inverse-Hessian mass matrix + MAP init -- skips the
        # expensive early-warmup transient. Isolated from the standard
        # model-based path below (which is unchanged).
        if self.config.precondition == 'laplace' or self._preconditioner is not None:
            return self._run_preconditioned()

        start_time = time.time()

        # Setup random key
        seed = self.config.seed if self.config.seed is not None else int(time.time())
        rng_key = random.PRNGKey(seed)
        rng_key, init_key, sample_key = random.split(rng_key, 3)

        # Compute reparameterization scales
        self._reparam_scales = self._compute_reparam_scales()

        # Build model
        model = self._build_numpyro_model(self._reparam_scales)

        # Setup NUTS kernel
        kernel = NUTS(
            model,
            dense_mass=self.config.dense_mass,
            max_tree_depth=self.config.max_tree_depth,
            target_accept_prob=self.config.target_accept_prob,
        )

        chain_method = self._resolve_chain_method(self.config.n_chains)

        # Setup MCMC
        mcmc = MCMC(
            kernel,
            num_warmup=self.config.n_warmup,
            num_samples=self.config.n_samples,
            num_chains=self.config.n_chains,
            chain_method=chain_method,
            progress_bar=self.config.progress,
        )

        # Get initial parameters
        init_params = self._get_init_params(init_key, self._reparam_scales)

        # Run MCMC
        mcmc.run(
            sample_key,
            init_params=init_params,
            extra_fields=(
                'diverging',
                'accept_prob',
                'num_steps',
                'energy',
            ),
        )

        # Extract samples (physical space via deterministic nodes)
        samples_dict = mcmc.get_samples(group_by_chain=False)

        # Build samples array in sampled_names order
        n_total_samples = self.config.n_samples * self.config.n_chains
        samples_list = []
        for name in self.task.sampled_names:
            # the model registers a deterministic site (physical space) for
            # every sampled name, so it is always present; assert rather than
            # silently reconstructing from the _z_ latents.
            assert name in samples_dict, (
                f"expected deterministic site '{name}' missing from numpyro "
                f"samples; model construction changed?"
            )
            samples_list.append(np.array(samples_dict[name]).flatten())

        samples = np.column_stack(samples_list)

        # Compute log probabilities for samples (chunked to bound peak memory;
        # see _batched_log_posterior_chunked).
        log_probs = _batched_log_posterior_chunked(
            self.task._log_posterior_jittable, samples
        )

        # Collect diagnostics
        diagnostics = self._collect_diagnostics(mcmc, self._reparam_scales)

        # Compute acceptance fraction
        acceptance_fraction = diagnostics.get('mean_accept_prob', None)

        # Check convergence
        r_hats = diagnostics.get('r_hat', {})
        max_rhat = max(r_hats.values()) if r_hats else 1.0
        converged = max_rhat < 1.1 and diagnostics.get('divergence_rate', 0) < 0.1

        # Build metadata
        elapsed = time.time() - start_time
        metadata = {
            'sampler': 'numpyro',
            'algorithm': 'nuts',
            'elapsed_seconds': elapsed,
            'n_chains': self.config.n_chains,
            'n_warmup': self.config.n_warmup,
            'n_samples_per_chain': self.config.n_samples,
            'seed': seed,
            'dense_mass': self.config.dense_mass,
            'reparam_strategy': self.config.reparam_strategy.value,
            'chain_method': chain_method,
        }

        return SamplerResult(
            samples=samples,
            log_prob=log_probs,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            acceptance_fraction=acceptance_fraction,
            converged=converged,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _run_preconditioned(self) -> SamplerResult:
        """Laplace-preconditioned NUTS (opt-in via ``precondition='laplace'``).

        Computes (or reuses) a Laplace preconditioner -- MAP + regularized
        inverse Hessian -- then runs physical-space ``potential_fn`` NUTS with
        that as a FIXED mass matrix, initialized at the MAP. This skips the
        expensive early-warmup transient (an identity-metric chain climbing
        from scratch) and the dense-mass adaptation cost. ~2x faster than
        adapted dense mass with better convergence on the flagship joint
        test. The standard model-based ``run`` path is untouched.
        """
        from numpyro.infer import MCMC, NUTS

        from kl_pipe.sampling.initialization import chain_inits

        start_time = time.time()
        seed = self.config.seed if self.config.seed is not None else int(time.time())

        pre = self._preconditioner
        if pre is None:
            pre = self.task.laplace_preconditioner(
                n_starts=self.config.n_map_starts,
                seed=seed,
                hessian_method=self.config.hessian_method,
            )
            self._preconditioner = pre

        sampled_names = self.task.sampled_names
        n_params = len(sampled_names)
        log_posterior_fn = self.task.get_log_posterior_fn()

        # optional unconstrained sampling coordinates: bijections chosen from
        # prior support bounds remove the -inf truncation walls (each NUTS
        # trajectory crossing a wall is recorded as a divergence); the
        # physical-space posterior is unchanged by construction
        transform = None
        clipped_names: list = []
        if self.config.precondition_unconstrained:
            from kl_pipe.sampling.transforms import UnconstrainingTransform

            transform = UnconstrainingTransform.from_priors(self.task.priors)

        if transform is None:

            def potential_fn(theta):
                return -log_posterior_fn(theta)

            inv_mass = jnp.asarray(pre.inverse_mass_matrix)
            theta_map = jnp.asarray(pre.map_point)
        else:

            def potential_fn(eta):
                theta = transform.inverse(eta)
                return -log_posterior_fn(theta) - transform.log_jacobian(eta)

            eta_map, clipped = transform.forward_clipped(
                np.asarray(pre.map_point), u_margin=1e-6
            )
            if clipped.any():
                clipped_names = [
                    sampled_names[i] for i in np.where(clipped)[0].tolist()
                ]
                warnings.warn(
                    f"MAP point sits on/outside the open prior support for "
                    f"{clipped_names}; clipped into the support margin for "
                    f"the NUTS init (the sampler itself is unaffected)."
                )
            inv_mass = jnp.asarray(
                transform.transform_inverse_mass(pre.inverse_mass_matrix, eta_map)
            )
            theta_map = jnp.asarray(eta_map)

        # optional donated metric: an explicit initial inverse mass matrix in
        # sampling coordinates (e.g. a previous same-fit run's warmup-adapted
        # metric) replaces the (transformed) Laplace metric
        if self.config.init_inverse_mass_matrix is not None:
            donated = np.asarray(self.config.init_inverse_mass_matrix)
            if donated.shape != (n_params, n_params):
                raise ValueError(
                    f"init_inverse_mass_matrix shape {donated.shape} does not "
                    f"match the task's sampled dimension ({n_params}, "
                    f"{n_params})"
                )
            inv_mass = jnp.asarray(donated)

        # chain initial points in sampling coordinates: all at the MAP with a
        # 1% metric-scale jitter, or one chain per competing MAP basin
        n_chains = self.config.n_chains
        init_params = jnp.asarray(
            chain_inits(
                pre,
                np.asarray(inv_mass),
                n_chains,
                mode=self.config.chain_init,
                seed=seed,
                transform=transform,
                max_margin=self.config.chain_init_max_margin,
            )
        )
        if n_chains == 1:
            init_params = init_params[0]

        kernel = NUTS(
            potential_fn=potential_fn,
            dense_mass=True,
            inverse_mass_matrix=inv_mass,
            # default: fixed Laplace metric (validated recipe); opt-in warmup
            # adaptation starting FROM that metric for posteriors whose soft
            # directions the MAP Hessian mis-measures
            adapt_mass_matrix=self.config.precondition_adapt_mass,
            adapt_step_size=True,
            max_tree_depth=self.config.max_tree_depth,
            target_accept_prob=self.config.target_accept_prob,
        )
        chain_method = self._resolve_chain_method(n_chains)

        mcmc = MCMC(
            kernel,
            num_warmup=self.config.n_warmup,
            num_samples=self.config.n_samples,
            num_chains=n_chains,
            chain_method=chain_method,
            progress_bar=self.config.progress,
        )
        mcmc.run(
            random.PRNGKey(seed + 1),
            init_params=init_params,
            extra_fields=('diverging', 'accept_prob', 'num_steps', 'energy'),
        )

        # potential_fn samples come back as a flat array in sampled_names
        # order; back-transform to physical coordinates if the chain ran in
        # unconstrained space.
        samples = np.asarray(mcmc.get_samples())
        grouped = np.asarray(mcmc.get_samples(group_by_chain=True))
        if transform is not None:
            samples = np.asarray(transform.inverse(samples))
            grouped = np.asarray(transform.inverse(grouped))
            if transform.is_periodic.any():
                # periodic dims: one contiguous branch centred on the MAP so
                # the linear r-hat, ESS, mean and std describe the posterior
                samples = transform.wrap_about(samples, pre.map_point)
                grouped = transform.wrap_about(grouped, pre.map_point)

        log_probs = _batched_log_posterior_chunked(
            self.task._log_posterior_jittable, samples
        )

        samples_by_chain = {
            name: grouped[:, :, i] for i, name in enumerate(sampled_names)
        }
        diagnostics = self._collect_diagnostics(
            mcmc, reparam_scales=None, samples_by_chain=samples_by_chain
        )
        diagnostics['preconditioner'] = {
            'method': 'laplace',
            'condition_number': pre.condition_number,
            'n_starts_converged': pre.n_starts_converged,
            'n_negative_eigenvalues': pre.n_negative_eigenvalues,
            'min_eigenvalue_ratio': pre.min_eigenvalue_ratio,
        }
        if self.config.precondition_adapt_mass:
            # final warmup-adapted metric (sampling coordinates): reusable as
            # a donor mass matrix for sibling fits and escalation reruns
            diagnostics['adapted_inverse_mass_matrix'] = np.asarray(
                mcmc.last_state.adapt_state.inverse_mass_matrix
            )
        if transform is not None:
            diagnostics['preconditioner']['unconstrained'] = {
                'kinds': dict(zip(sampled_names, transform.kind_names)),
                'map_clipped_params': clipped_names,
            }

        acceptance_fraction = diagnostics.get('mean_accept_prob', None)
        r_hats = diagnostics.get('r_hat', {})
        max_rhat = max(r_hats.values()) if r_hats else 1.0
        converged = max_rhat < 1.1 and diagnostics.get('divergence_rate', 0) < 0.1

        elapsed = time.time() - start_time
        metadata = {
            'sampler': 'numpyro',
            'algorithm': 'nuts',
            'elapsed_seconds': elapsed,
            'n_chains': n_chains,
            'n_warmup': self.config.n_warmup,
            'n_samples_per_chain': self.config.n_samples,
            'seed': seed,
            'dense_mass': True,
            'precondition': 'laplace',
            'precondition_unconstrained': transform is not None,
            'init_mass_donated': self.config.init_inverse_mass_matrix is not None,
            'chain_init': self.config.chain_init,
            'chain_method': chain_method,
        }
        result = SamplerResult(
            samples=samples,
            log_prob=log_probs,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            acceptance_fraction=acceptance_fraction,
            converged=converged,
            diagnostics=diagnostics,
            metadata=metadata,
        )
        # warm state kept for continue_sampling (more draws, no re-warmup)
        self._continuation = {
            'mcmc': mcmc,
            'transform': transform,
            'map_point': np.asarray(pre.map_point),
            'grouped': grouped,
            'result': result,
            'n_chains': n_chains,
            'chain_method': chain_method,
        }
        return result

    def continue_sampling(self, n_samples: int) -> SamplerResult:
        """Draw ``n_samples`` more per chain from the completed preconditioned
        run's final state (position, step size, metric), with no warmup, and
        return the union of all draws so far as one result.

        The returned samples are chain-major (all draws of chain 0, then
        chain 1, ...), r-hat/ESS are recomputed on the combined per-chain
        draws, and the per-draw diagnostic arrays (divergences, acceptance,
        leapfrog steps) are the first run's followed by the new draws'.
        Repeatable: each call extends the same chains.
        """
        from numpyro.infer import MCMC

        if getattr(self, '_continuation', None) is None:
            raise RuntimeError(
                "continue_sampling requires a completed preconditioned run "
                "(precondition='laplace') on this sampler instance"
            )
        if (
            not isinstance(n_samples, int)
            or isinstance(n_samples, bool)
            or n_samples < 1
        ):
            raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")
        c = self._continuation
        start_time = time.time()
        prev = c['result']
        sampled_names = list(self.task.sampled_names)
        n_params = len(sampled_names)

        # one no-warmup MCMC object per block size, reused across blocks
        block_mcmcs = c.setdefault('block_mcmcs', {})
        mcmc = block_mcmcs.get(n_samples)
        if mcmc is None:
            mcmc = MCMC(
                c['mcmc'].sampler,
                num_warmup=0,
                num_samples=n_samples,
                num_chains=c['n_chains'],
                chain_method=c['chain_method'],
                progress_bar=self.config.progress,
            )
            block_mcmcs[n_samples] = mcmc
        mcmc.post_warmup_state = c['mcmc'].last_state
        mcmc.run(
            c['mcmc'].last_state.rng_key,
            extra_fields=('diverging', 'accept_prob', 'num_steps', 'energy'),
        )

        grouped_new = np.asarray(mcmc.get_samples(group_by_chain=True))
        grouped_new = grouped_new.reshape(c['n_chains'], n_samples, n_params)
        transform = c['transform']
        if transform is not None:
            grouped_new = np.asarray(transform.inverse(grouped_new))
            if transform.is_periodic.any():
                grouped_new = transform.wrap_about(grouped_new, c['map_point'])
        grouped = np.concatenate([c['grouped'], grouped_new], axis=1)
        samples = grouped.reshape(-1, n_params)
        log_probs_new = _batched_log_posterior_chunked(
            self.task._log_posterior_jittable, grouped_new.reshape(-1, n_params)
        )
        n_prev = c['grouped'].shape[1]
        log_probs = np.concatenate(
            [
                np.asarray(prev.log_prob).reshape(c['n_chains'], n_prev),
                log_probs_new.reshape(c['n_chains'], n_samples),
            ],
            axis=1,
        ).reshape(-1)

        samples_by_chain = {
            name: grouped[:, :, i] for i, name in enumerate(sampled_names)
        }
        diagnostics = self._collect_diagnostics(
            mcmc, reparam_scales=None, samples_by_chain=samples_by_chain
        )
        # per-draw arrays: first run's draws followed by the new draws
        for key in ('diverging', 'accept_prob', 'num_steps'):
            old = prev.diagnostics.get(key)
            new = diagnostics.get(key)
            if old is None or new is None:
                raise RuntimeError(f"continue_sampling: per-draw field '{key}' missing")
            diagnostics[key] = np.concatenate([np.asarray(old), np.asarray(new)])
        diverging = diagnostics['diverging']
        diagnostics['n_divergences'] = int(diverging.sum())
        diagnostics['divergence_rate'] = (
            float(diverging.mean()) if diverging.size else 0.0
        )
        diagnostics['mean_accept_prob'] = float(np.mean(diagnostics['accept_prob']))
        diagnostics['mean_tree_depth'] = float(
            np.log2(np.asarray(diagnostics['num_steps']) + 1).mean()
        )
        diagnostics['preconditioner'] = dict(prev.diagnostics['preconditioner'])
        if self.config.precondition_adapt_mass:
            diagnostics['adapted_inverse_mass_matrix'] = np.asarray(
                mcmc.last_state.adapt_state.inverse_mass_matrix
            )

        r_hats = diagnostics.get('r_hat', {})
        max_rhat = max(r_hats.values()) if r_hats else 1.0
        converged = max_rhat < 1.1 and diagnostics['divergence_rate'] < 0.1
        metadata = dict(prev.metadata)
        metadata['elapsed_seconds'] = float(prev.metadata['elapsed_seconds']) + (
            time.time() - start_time
        )
        metadata['n_samples_per_chain'] = int(grouped.shape[1])
        metadata['continuations'] = int(prev.metadata.get('continuations', 0)) + 1
        metadata['continued_draws_per_chain'] = int(
            prev.metadata.get('continued_draws_per_chain', 0)
        ) + int(n_samples)
        result = SamplerResult(
            samples=samples,
            log_prob=log_probs,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            acceptance_fraction=diagnostics['mean_accept_prob'],
            converged=converged,
            diagnostics=diagnostics,
            metadata=metadata,
        )
        self._continuation = dict(c, mcmc=mcmc, grouped=grouped, result=result)
        return result
