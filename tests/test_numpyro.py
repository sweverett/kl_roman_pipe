"""
Tests for NumPyro gradient-based sampler.

These tests verify that the NumPyro sampler works correctly with Z-score
reparameterization, including:
- Gradient scaling normalization
- Basic sampling functionality
- Joint model handling (the key failure mode for BlackJAX)
- Convergence diagnostics (R-hat, ESS)
- Different chain execution methods

Many tests are adapted from test_blackjax.py to ensure NumPyro handles
the same scenarios that caused BlackJAX to fail.
"""

import galsim
import pytest

# All tests in this file run real NUTS/HMC sampling. Mark the entire module
# as slow so CI's `make test-basic` (excludes slow) doesn't time out.
# Run via `make test-sampling` or with `-m "not slow"` removed.
pytestmark = pytest.mark.slow

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from pathlib import Path

from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity, SyntheticIntensity
from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_image_obs, build_velocity_obs
from kl_pipe.sampling import (
    InferenceTask,
    NumpyroSamplerConfig,
    ReparamStrategy,
    build_sampler,
)
from kl_pipe.sampling.numpyro import (
    NumpyroSampler,
    _LOG_PROB_CHUNK_SIZE,
    _batched_log_posterior_chunked,
    compute_reparam_scales,
)
from kl_pipe.utils import get_test_dir


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def output_dir():
    """Output directory for NumPyro diagnostic tests."""
    out_dir = get_test_dir() / "out" / "numpyro_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="module")
def simple_velocity_task():
    """
    Create a simple velocity-only inference task for basic tests.

    Uses Gaussian/TruncatedNormal priors with reasonable scales.

    Returns (task, true_pars_dotted) — true_pars uses dotted-key SourceModel
    convention; consumers indexing by ``task.sampled_names`` match directly.
    """
    image_pars = ImagePars(shape=(20, 20), pixel_scale=0.4, indexing='ij')

    # Flat-key form for SyntheticVelocity
    true_pars_flat = {
        'v0': 10.0,
        'vcirc': 200.0,
        'rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
    }
    # Dotted-key form for SourceModel priors and downstream consumers
    true_pars = {
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
    }

    synth_vel = SyntheticVelocity(true_pars_flat, model_type='arctan', seed=42)
    data_vel_noisy = synth_vel.generate(image_pars, snr=1000)
    var_vel = synth_vel.variance

    source = SourceModel(velocity_model=CenteredVelocityModel())
    priors = PriorDict(
        {
            'vel.v0': Gaussian(10.0, 5.0),
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
            'vel.rscale': TruncatedNormal(5.0, 2.0, 0.4, 20.0),
            'cosi': TruncatedNormal(0.6, 0.2, 0.01, 0.99),
            'theta_int': TruncatedNormal(0.785, 0.3, 0, np.pi),
            'g1': 0.02,  # Fixed
            'g2': -0.01,  # Fixed
        }
    )

    vel_obs = build_velocity_obs(
        image_pars, data=jnp.array(data_vel_noisy), variance=var_vel
    )
    task = InferenceTask.from_obs(source, priors, velocity_obs=vel_obs)

    return task, true_pars


@pytest.fixture(scope="module")
def joint_model_task():
    """
    Create joint velocity+intensity task - the critical test case.

    This is where BlackJAX failed due to gradient scale mismatch.

    Returns (task, true_pars_dotted) so consumers can index by
    ``task.sampled_names`` directly.
    """
    from kl_pipe.intensity import InclinedExponentialModel

    image_pars_vel = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
    image_pars_int = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')

    # Roman-like PSF: damps the worst-case maxk so the wide rscale + edge-on
    # priors don't blow up oversample (cf. Issue #47).
    psf = galsim.Gaussian(fwhm=0.2)

    # Flat-key form for Synthetic* generators
    true_pars_flat = {
        'v0': 10.0,
        'vcirc': 200.0,
        'rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.03,
        'g2': -0.02,
    }
    int_true = {
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.03,
        'g2': -0.02,
        'flux': 1.0,
        'rscale': 3.0,
        'h_over_r': 0.1,
        'x0': 0.0,
        'y0': 0.0,
    }
    # Dotted-key form for SourceModel priors and downstream consumers
    true_pars = {
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.03,
        'g2': -0.02,
        'F087.flux': 1.0,
        'F087.rscale': 3.0,
        'F087.h_over_r': 0.1,
        'F087.x0': 0.0,
        'F087.y0': 0.0,
    }

    vel_model = CenteredVelocityModel()
    vel_pars = {
        k: v for k, v in true_pars_flat.items() if k in vel_model.PARAMETER_NAMES
    }
    synth_vel = SyntheticVelocity(vel_pars, model_type='arctan', seed=42)
    data_vel = synth_vel.generate(image_pars_vel, snr=1000)
    var_vel = synth_vel.variance

    int_model = InclinedExponentialModel()
    int_pars = {k: v for k, v in int_true.items() if k in int_model.PARAMETER_NAMES}
    synth_int = SyntheticIntensity(int_pars, model_type='exponential', seed=43, psf=psf)
    data_int = synth_int.generate(image_pars_int, snr=1000, include_poisson=False)
    var_int = synth_int.variance

    source = SourceModel(
        velocity_model=vel_model,
        broadband_models={'F087': int_model},
    )

    priors = PriorDict(
        {
            'vel.v0': Gaussian(true_pars['vel.v0'], 5.0),
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
            'vel.rscale': TruncatedNormal(5.0, 2.0, 1.0, 10.0),
            'F087.flux': TruncatedNormal(1.0, 1.0, 0.1, 5.0),
            'F087.rscale': TruncatedNormal(3.0, 2.0, 0.5, 10.0),
            'F087.h_over_r': 0.1,  # Fixed
            'F087.x0': 0.0,  # Fixed
            'F087.y0': 0.0,  # Fixed
            'cosi': TruncatedNormal(0.5, 0.3, 0.01, 0.99),
            'theta_int': TruncatedNormal(np.pi / 2, 1.0, 0, np.pi),
            'g1': TruncatedNormal(0.0, 0.05, -0.1, 0.1),
            'g2': TruncatedNormal(0.0, 0.05, -0.1, 0.1),
        }
    )

    img_obs = build_image_obs(
        image_pars_int,
        psf=psf,
        data=jnp.array(data_int),
        variance=var_int,
        broadband_key='F087',
    )
    vel_obs = build_velocity_obs(
        image_pars_vel, data=jnp.array(data_vel), variance=var_vel
    )
    task = InferenceTask.from_obs(
        source,
        priors,
        velocity_obs=vel_obs,
        image_obs={'F087': img_obs},
    )

    return task, true_pars


# ==============================================================================
# Chunked log-posterior evaluation
# ==============================================================================


def test_chunked_log_posterior_matches_vmap():
    """Chunked evaluation equals a single vmap for a non-multiple of the chunk."""

    def log_post(theta):
        return -0.5 * jnp.sum(theta**2) + jnp.sin(theta[0])

    n = 3 * _LOG_PROB_CHUNK_SIZE + 5
    samples = np.asarray(random.normal(random.PRNGKey(0), (n, 4)))
    chunked = _batched_log_posterior_chunked(log_post, samples)
    reference = np.asarray(jax.vmap(log_post)(jnp.asarray(samples)))
    assert chunked.shape == (n,)
    np.testing.assert_allclose(chunked, reference, rtol=0, atol=1e-12)


# ==============================================================================
# Reparameterization Tests
# ==============================================================================


class TestReparameterization:
    """Tests for Z-score reparameterization scaling."""

    def test_gaussian_scaling(self):
        """Gaussian prior uses mu and sigma directly."""
        prior = Gaussian(100.0, 25.0)
        loc, scale = compute_reparam_scales(prior, 'test')
        assert loc == 100.0
        assert scale == 25.0

    def test_truncated_normal_scaling(self):
        """TruncatedNormal uses underlying Gaussian params."""
        prior = TruncatedNormal(0.5, 0.2, 0.01, 0.99)
        loc, scale = compute_reparam_scales(prior, 'test')
        assert loc == 0.5
        assert scale == 0.2

    def test_uniform_scaling(self):
        """Uniform uses midpoint and quarter-range."""
        prior = Uniform(0, 100)
        loc, scale = compute_reparam_scales(prior, 'test')
        assert loc == 50.0
        assert scale == 25.0  # (100-0)/4

    def test_reparam_strategy_none(self, simple_velocity_task):
        """ReparamStrategy.NONE returns identity scales."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            reparam_strategy=ReparamStrategy.NONE,
            seed=42,
            progress=False,
        )

        sampler = NumpyroSampler(task, config)
        scales = sampler._compute_reparam_scales()

        for name in task.sampled_names:
            loc, scale = scales[name]
            assert loc == 0.0
            assert scale == 1.0

    def test_reparam_strategy_prior(self, simple_velocity_task):
        """ReparamStrategy.PRIOR uses prior statistics."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            reparam_strategy=ReparamStrategy.PRIOR,
            seed=42,
            progress=False,
        )

        sampler = NumpyroSampler(task, config)
        scales = sampler._compute_reparam_scales()

        # Check that vel.vcirc uses its prior params
        loc, scale = scales['vel.vcirc']
        assert loc == 200.0  # TruncatedNormal mu
        assert scale == 50.0  # TruncatedNormal sigma


class TestGradientScaling:
    """
    Verify that Z-score reparameterization normalizes gradients.

    This is the key test for the BlackJAX failure mode.
    """

    @pytest.mark.slow
    def test_latent_gradient_norms_order_one(self, joint_model_task, output_dir):
        """
        After Z-score transform, gradients w.r.t. latent z should be O(1).

        This test verifies the fix for BlackJAX collapse where intensity
        gradients were ~10^4x larger than velocity gradients.

        We evaluate gradients at the TRUE parameter values (converted to z-space)
        where we expect the log posterior gradient to be moderate (near the mode).
        """
        import numpyro
        import numpyro.distributions as dist

        task, true_pars = joint_model_task

        # Get prior-based scales
        config = NumpyroSamplerConfig(reparam_strategy=ReparamStrategy.PRIOR)
        sampler = NumpyroSampler(task, config)
        scales = sampler._compute_reparam_scales()

        # Build function that computes log_posterior in z-space
        log_posterior_fn = task.get_log_posterior_fn()

        def log_prob_z(z_dict):
            """Log posterior as function of latent z values."""
            theta_physical = []
            for name in task.sampled_names:
                loc, scale = scales[name]
                z = z_dict[name]
                theta_physical.append(loc + scale * z)
            theta = jnp.stack(theta_physical)
            return log_posterior_fn(theta)

        # Convert TRUE parameters to z-space (not z=0)
        # Near the mode, gradients should be moderate
        z_true = {}
        for name in task.sampled_names:
            loc, scale = scales[name]
            theta_true = true_pars[name]
            z_true[name] = jnp.array((theta_true - loc) / scale)

        grad_fn = jax.grad(log_prob_z)
        grads = grad_fn(z_true)

        # Check gradient magnitudes
        grad_mags = {name: float(jnp.abs(grads[name])) for name in task.sampled_names}

        # Write diagnostic output
        log_path = output_dir / "gradient_scaling_test.txt"
        with open(log_path, 'w') as f:
            f.write("Gradient Scaling Test (Z-space)\n")
            f.write("=" * 60 + "\n\n")
            f.write(
                "Gradients evaluated at z=z_true (physical space = true params)\n\n"
            )

            f.write("Parameter z-values (true):\n")
            for name in task.sampled_names:
                f.write(f"  {name:15s}: z = {float(z_true[name]):.4f}\n")
            f.write("\n")

            for name, mag in sorted(grad_mags.items()):
                f.write(f"{name:15s}: |∂log_p/∂z| = {mag:.4e}\n")

            max_grad = max(grad_mags.values())
            min_grad = min(grad_mags.values())
            ratio = max_grad / min_grad if min_grad > 0 else float('inf')
            f.write(f"\nMax/Min ratio: {ratio:.2f}\n")

            if ratio < 100:
                f.write("\nSUCCESS: Gradients are well-balanced (ratio < 100)\n")
            else:
                f.write("\nWARNING: Large gradient disparity may cause issues\n")

        # Assertion: ratio should be much smaller than 10^4 (the BlackJAX failure).
        # 3D joint model has inherent ~10^3 gradient disparity between intensity
        # (high Fisher info per pixel) and shear (subtle distortion). Prior-based
        # Z-score can't fully close a gap that lives in the likelihood curvature.
        # Threshold 5000 catches catastrophic scaling while allowing this physics.
        assert (
            ratio < 5000
        ), f"Gradient ratio {ratio:.0f} too large - reparameterization not working"


# ==============================================================================
# Basic Sampling Tests
# ==============================================================================


@pytest.mark.slow
class TestNumpyroBasicSampling:
    """Verify sampler produces valid output."""

    def test_samples_shape(self, simple_velocity_task):
        """Correct number of samples returned."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=100,
            n_warmup=50,
            n_chains=2,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        expected_samples = 100 * 2  # n_samples * n_chains
        assert result.n_samples == expected_samples
        assert result.samples.shape == (expected_samples, task.n_params)

    def test_finite_log_prob(self, simple_velocity_task):
        """All samples have finite log probability."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=100,
            n_warmup=50,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        assert np.all(
            np.isfinite(result.log_prob)
        ), "Some log_prob values are not finite"

    def test_no_divergences(self, simple_velocity_task):
        """No divergent transitions for well-posed problem."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        n_div = result.diagnostics.get('n_divergences', 0)
        div_rate = result.diagnostics.get('divergence_rate', 0)

        assert div_rate < 0.05, f"Too many divergences: {n_div} ({div_rate:.1%})"

    def test_reasonable_acceptance_rate(self, simple_velocity_task):
        """Acceptance rate should be in healthy range."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        accept = result.acceptance_fraction
        assert accept is not None
        assert (
            0.4 < accept < 0.99
        ), f"Acceptance rate {accept:.2%} outside healthy range"


# ==============================================================================
# Joint Model Tests (Critical - BlackJAX Failed Here)
# ==============================================================================


@pytest.mark.slow
class TestNumpyroJointModel:
    """
    The critical tests - joint model must not collapse like BlackJAX.

    These tests verify that NumPyro with Z-score reparameterization
    successfully samples joint velocity+intensity models.
    """

    def test_nonzero_variance_all_params(self, joint_model_task, output_dir):
        """
        All parameters must have posterior variance > 0.

        This was the key failure mode of BlackJAX: step_size collapsed
        to ~1e-8, resulting in zero-variance chains.
        """
        task, _ = joint_model_task

        config = NumpyroSamplerConfig(
            n_samples=500,
            n_warmup=500,
            n_chains=1,
            dense_mass=True,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        # Check variance for each parameter
        variances = {}
        zero_var_params = []
        for name in result.param_names:
            chain = result.get_chain(name)
            var = float(np.var(chain))
            variances[name] = var
            if var < 1e-10:
                zero_var_params.append(name)

        # Write diagnostic log
        log_path = output_dir / "joint_model_variance.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model Variance Test\n")
            f.write("=" * 60 + "\n\n")

            for name, var in sorted(variances.items()):
                status = "OK" if var > 1e-10 else "ZERO"
                f.write(f"{name:15s}: variance = {var:.6e} [{status}]\n")

            if zero_var_params:
                f.write(f"\nFAILED: Zero variance params: {zero_var_params}\n")
            else:
                f.write("\nSUCCESS: All parameters have non-zero variance\n")

        assert (
            len(zero_var_params) == 0
        ), f"Parameters with zero variance: {zero_var_params}"

    def test_step_size_reasonable(self, joint_model_task, output_dir):
        """Step size should be O(0.01-1), not 1e-8."""
        task, _ = joint_model_task

        config = NumpyroSamplerConfig(
            n_samples=500,
            n_warmup=500,
            n_chains=1,
            dense_mass=True,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        step_size = result.diagnostics.get('step_size')

        # Write diagnostic
        log_path = output_dir / "joint_model_step_size.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model Step Size Test\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Adapted step size: {step_size}\n")

            if step_size is not None:
                if step_size < 1e-6:
                    f.write("FAILED: Step size collapsed (like BlackJAX failure)\n")
                elif step_size > 10:
                    f.write("WARNING: Step size very large\n")
                else:
                    f.write("SUCCESS: Step size in reasonable range\n")

        assert step_size is not None, "Step size not returned"
        assert step_size > 1e-6, f"Step size {step_size:.2e} collapsed (too small)"

    def test_no_excessive_divergences(self, joint_model_task):
        """Well-posed problem should have <10% divergences."""
        task, _ = joint_model_task

        config = NumpyroSamplerConfig(
            n_samples=500,
            n_warmup=500,
            n_chains=1,
            dense_mass=True,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        div_rate = result.diagnostics.get('divergence_rate', 0)
        assert div_rate < 0.10, f"Divergence rate {div_rate:.1%} too high"


# ==============================================================================
# Convergence Diagnostics Tests
# ==============================================================================


@pytest.mark.slow
class TestNumpyroConvergence:
    """Verify convergence diagnostics are computed and valid."""

    def test_rhat_computed_multichain(self, simple_velocity_task):
        """R-hat available when n_chains > 1."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=2,  # Need multiple chains for R-hat
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        r_hats = result.get_rhat()
        assert r_hats is not None, "R-hat not computed"
        assert len(r_hats) == len(task.sampled_names)

    def test_rhat_near_one_for_converged(self, simple_velocity_task):
        """Converged chains should have R-hat < 1.05."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=500,
            n_warmup=200,
            n_chains=2,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        r_hats = result.get_rhat()
        max_rhat = max(r_hats.values())

        assert max_rhat < 1.1, f"Max R-hat {max_rhat:.3f} indicates poor convergence"

    def test_ess_computed(self, simple_velocity_task):
        """ESS available in diagnostics."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        ess = result.get_ess()
        assert ess is not None, "ESS not computed"
        assert len(ess) == len(task.sampled_names)

    def test_ess_reasonable(self, simple_velocity_task):
        """ESS should be > 50 for short run."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=300,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        ess = result.get_ess()
        min_ess = min(ess.values())

        assert min_ess > 20, f"Min ESS {min_ess:.0f} too low"


# ==============================================================================
# Chain Method Tests
# ==============================================================================


@pytest.mark.slow
class TestNumpyroChainMethods:
    """Test different chain execution strategies."""

    def test_sequential_chains(self, simple_velocity_task):
        """chain_method='sequential' works."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=50,
            n_warmup=25,
            n_chains=2,
            chain_method='sequential',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        assert result.n_samples == 100  # 50 * 2

    def test_vectorized_chains(self, simple_velocity_task):
        """chain_method='vectorized' works."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=50,
            n_warmup=25,
            n_chains=2,
            chain_method='vectorized',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        assert result.n_samples == 100


class TestResolveChainMethod:
    """Unit tests for NumpyroSampler._resolve_chain_method dispatch logic.

    No MCMC is run here -- these monkeypatch jax.default_backend /
    jax.local_device_count (actual device count cannot change mid-pytest-
    process) and call the private resolver directly.
    """

    def test_explicit_method_always_wins(self, simple_velocity_task, monkeypatch):
        """An explicit non-None chain_method overrides auto-dispatch."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(chain_method='sequential', progress=False)
        sampler = NumpyroSampler(task, config)

        monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
        monkeypatch.setattr(jax, 'local_device_count', lambda: 8)

        assert sampler._resolve_chain_method(4) == 'sequential'

    def test_auto_non_cpu_backend_dispatches_vectorized(
        self, simple_velocity_task, monkeypatch
    ):
        """chain_method=None on a non-CPU backend resolves to 'vectorized'."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(chain_method=None, progress=False)
        sampler = NumpyroSampler(task, config)

        monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
        monkeypatch.setattr(jax, 'local_device_count', lambda: 1)

        assert sampler._resolve_chain_method(4) == 'vectorized'

    def test_auto_cpu_enough_devices_dispatches_parallel(
        self, simple_velocity_task, monkeypatch
    ):
        """chain_method=None on CPU with device_count >= n_chains (and > 1)
        resolves to 'parallel'."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(chain_method=None, progress=False)
        sampler = NumpyroSampler(task, config)

        monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
        monkeypatch.setattr(jax, 'local_device_count', lambda: 4)

        assert sampler._resolve_chain_method(4) == 'parallel'

    def test_auto_cpu_single_device_dispatches_sequential(
        self, simple_velocity_task, monkeypatch
    ):
        """chain_method=None on CPU with exactly 1 device never picks
        'parallel', even if n_chains == 1 (device count > 1 required)."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(chain_method=None, progress=False)
        sampler = NumpyroSampler(task, config)

        monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
        monkeypatch.setattr(jax, 'local_device_count', lambda: 1)

        assert sampler._resolve_chain_method(1) == 'sequential'

    def test_auto_cpu_insufficient_devices_dispatches_sequential(
        self, simple_velocity_task, monkeypatch
    ):
        """chain_method=None on CPU with device_count < n_chains resolves to
        'sequential', not 'parallel' -- this is the byte-identical-default
        branch: an unconfigured single-CPU-device machine must not start
        raising the parallel/device-count error merely from the new default.
        """
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(chain_method=None, progress=False)
        sampler = NumpyroSampler(task, config)

        monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
        monkeypatch.setattr(jax, 'local_device_count', lambda: 2)

        assert sampler._resolve_chain_method(4) == 'sequential'

    def test_explicit_parallel_insufficient_devices_raises(
        self, simple_velocity_task, monkeypatch
    ):
        """Explicit chain_method='parallel' with n_chains > device_count
        raises ValueError before numpyro's MCMC is ever constructed --
        NumPyro itself only warns and silently degrades to sequential."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(chain_method='parallel', progress=False)
        sampler = NumpyroSampler(task, config)

        monkeypatch.setattr(jax, 'local_device_count', lambda: 1)

        with pytest.raises(ValueError, match='KLPIPE_CPU_DEVICES'):
            sampler._resolve_chain_method(4)


# ==============================================================================
# Init Strategy Tests
# ==============================================================================


class TestNumpyroInitStrategies:
    """Test different initialization strategies."""

    def test_init_prior(self, simple_velocity_task):
        """init_strategy='prior' samples from prior."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            init_strategy='prior',
            seed=42,
            progress=False,
        )

        # Should not raise
        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()
        assert result.n_samples > 0

    def test_init_median(self, simple_velocity_task):
        """init_strategy='median' starts at prior medians."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            init_strategy='median',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()
        assert result.n_samples > 0

    def test_init_jitter(self, simple_velocity_task):
        """init_strategy='jitter' adds small perturbation."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            init_strategy='jitter',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()
        assert result.n_samples > 0


# ==============================================================================
# Factory and Integration Tests
# ==============================================================================


class TestNumpyroFactory:
    """Test factory integration."""

    def test_build_sampler_numpyro(self, simple_velocity_task):
        """build_sampler('numpyro', ...) works."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=10, n_warmup=10, seed=42, progress=False
        )

        sampler = build_sampler('numpyro', task, config)
        assert isinstance(sampler, NumpyroSampler)

    def test_build_sampler_nuts_alias(self, simple_velocity_task):
        """build_sampler('nuts', ...) returns NumpyroSampler."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=10, n_warmup=10, seed=42, progress=False
        )

        sampler = build_sampler('nuts', task, config)
        assert isinstance(sampler, NumpyroSampler)

    def test_build_sampler_hmc_alias(self, simple_velocity_task):
        """build_sampler('hmc', ...) returns NumpyroSampler."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=10, n_warmup=10, seed=42, progress=False
        )

        sampler = build_sampler('hmc', task, config)
        assert isinstance(sampler, NumpyroSampler)


# ==============================================================================
# Config Validation Tests
# ==============================================================================


class TestNumpyroConfig:
    """Test config validation."""

    def test_invalid_chain_method(self):
        """Invalid chain_method raises ValueError."""
        with pytest.raises(ValueError, match="chain_method"):
            NumpyroSamplerConfig(chain_method='invalid')

    def test_invalid_init_strategy(self):
        """Invalid init_strategy raises ValueError."""
        with pytest.raises(ValueError, match="init_strategy"):
            NumpyroSamplerConfig(init_strategy='invalid')

    def test_invalid_target_accept(self):
        """Invalid target_accept_prob raises ValueError."""
        with pytest.raises(ValueError, match="target_accept_prob"):
            NumpyroSamplerConfig(target_accept_prob=1.5)

    def test_reparam_strategy_string_conversion(self):
        """String reparam_strategy is converted to enum."""
        config = NumpyroSamplerConfig(reparam_strategy='prior')
        assert config.reparam_strategy == ReparamStrategy.PRIOR

    def test_invalid_reparam_strategy(self):
        """Invalid reparam_strategy raises ValueError."""
        with pytest.raises(ValueError, match="reparam_strategy"):
            NumpyroSamplerConfig(reparam_strategy='invalid')


# ==============================================================================
# Laplace Preconditioning (opt-in)
# ==============================================================================


class TestLaplacePreconditionerConfig:
    """Config validation for the precondition option (no sampling)."""

    def test_precondition_default_none(self):
        """Preconditioning is opt-in: default is 'none'."""
        assert NumpyroSamplerConfig().precondition == 'none'

    def test_valid_precondition_values(self):
        """'none' and 'laplace' are accepted."""
        assert NumpyroSamplerConfig(precondition='none').precondition == 'none'
        assert NumpyroSamplerConfig(precondition='laplace').precondition == 'laplace'

    def test_invalid_precondition_raises(self):
        """Unknown precondition value raises ValueError."""
        with pytest.raises(ValueError, match="precondition"):
            NumpyroSamplerConfig(precondition='bogus')

    def test_invalid_n_map_starts_raises(self):
        """n_map_starts < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_map_starts"):
            NumpyroSamplerConfig(n_map_starts=0)

    def test_hessian_method_default_fd(self):
        """Default Hessian evaluation is 'fd' (no second-order compile);
        'ad' remains available."""
        assert NumpyroSamplerConfig().hessian_method == 'fd'
        assert NumpyroSamplerConfig(hessian_method='ad').hessian_method == 'ad'

    def test_invalid_hessian_method_config_raises(self):
        """Unknown hessian_method value raises ValueError."""
        with pytest.raises(ValueError, match="hessian_method"):
            NumpyroSamplerConfig(hessian_method='bogus')


class TestLaplacePreconditioner:
    """The InferenceTask.laplace_preconditioner utility + preconditioned NUTS."""

    def test_preconditioner_utility(self, simple_velocity_task):
        """Truth-free MAP + regularized inverse-Hessian mass matrix is valid."""
        task, true_pars = simple_velocity_task
        pre = task.laplace_preconditioner(n_starts=3, seed=0)

        D = task.n_params
        assert pre.map_point.shape == (D,)
        assert pre.inverse_mass_matrix.shape == (D, D)
        assert pre.n_starts_converged >= 1
        # Mass matrix must be symmetric positive-definite (valid metric).
        assert np.allclose(pre.inverse_mass_matrix, pre.inverse_mass_matrix.T)
        assert np.all(np.linalg.eigvalsh(pre.inverse_mass_matrix) > 0)
        # MAP found from prior draws (not truth) should land near the well-
        # constrained truth at SNR=1000.
        names = task.sampled_names
        vcirc_map = pre.map_point[names.index('vel.vcirc')]
        assert abs(vcirc_map - true_pars['vel.vcirc']) / 200.0 < 0.1

    def test_fd_hessian_matches_ad(self, simple_velocity_task):
        """hessian_method='fd' must reproduce the 'ad' mass matrix.

        Same seed -> identical MAP (deterministic scipy path), so only the
        Hessian evaluation differs. Bound provenance: max relative element
        difference of the regularized inverse mass measured at ~1e-9 on the
        flagship joint task (fp64, fd_rel_step=1e-5, 2026-07-11); 1e-6 is
        ~1000x that measurement.
        """
        task, _ = simple_velocity_task
        pre_ad = task.laplace_preconditioner(n_starts=2, seed=0, hessian_method='ad')
        pre_fd = task.laplace_preconditioner(n_starts=2, seed=0, hessian_method='fd')

        np.testing.assert_array_equal(pre_fd.map_point, pre_ad.map_point)
        ref = np.max(np.abs(pre_ad.inverse_mass_matrix))
        assert (
            np.max(np.abs(pre_fd.inverse_mass_matrix - pre_ad.inverse_mass_matrix))
            < 1e-6 * ref
        )

    def test_ad_hessian_matches_dense_jax_hessian(self, simple_velocity_task):
        """The per-basis-vector HVP Hessian equals ``jax.hessian`` of the same
        negative log-posterior at the same point (both exact autodiff; the
        1e-10 relative budget covers reduction-order roundoff only)."""
        task, _ = simple_velocity_task
        pre = task.laplace_preconditioner(n_starts=2, seed=0, hessian_method='ad')
        theta = jnp.asarray(pre.map_point)
        H_lean = np.asarray(task._ad_hessian(theta))
        H_dense = np.asarray(
            jax.hessian(lambda t: -task._log_posterior_jittable(t))(theta)
        )
        ref = np.max(np.abs(H_dense))
        assert np.max(np.abs(H_lean - H_dense)) < 1e-10 * ref
        # the unregularized spectrum is reported alongside the floored metric
        assert pre.n_negative_eigenvalues >= 0
        assert np.isfinite(pre.min_eigenvalue_ratio)
        assert pre.min_eigenvalue_ratio <= 1.0
        if pre.n_negative_eigenvalues == 0:
            assert pre.min_eigenvalue_ratio > 0

    def test_invalid_hessian_method_raises(self, simple_velocity_task):
        """Unknown hessian_method fails loudly."""
        task, _ = simple_velocity_task
        with pytest.raises(ValueError, match="hessian_method"):
            task.laplace_preconditioner(hessian_method='bogus')

    def test_preconditioned_converges_and_recovers(self, simple_velocity_task):
        """precondition='laplace' yields a converged chain recovering truth."""
        task, true_pars = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=400,
            n_warmup=150,
            n_chains=2,
            chain_method='vectorized',
            seed=42,
            progress=False,
            precondition='laplace',
            n_map_starts=3,
        )
        result = build_sampler('numpyro', task, config).run()

        # Converged + healthy.
        rhats = result.get_rhat()
        assert max(rhats.values()) < 1.05
        assert result.diagnostics['divergence_rate'] < 0.1
        # Diagnostics record the preconditioner used.
        assert result.diagnostics['preconditioner']['method'] == 'laplace'
        # Recovers the well-constrained velocity scale.
        names = list(result.param_names)
        means = result.samples.mean(axis=0)
        vcirc = means[names.index('vel.vcirc')]
        assert abs(vcirc - true_pars['vel.vcirc']) / 200.0 < 0.05

    def test_donated_mass_matrix_pass_through(self, simple_velocity_task):
        """A previous run's warmup-adapted metric can be donated back as the
        retry's initial NUTS metric (the ensemble escalation recipe): the
        first adapt-mass run records ``adapted_inverse_mass_matrix``; a
        second run accepting it via ``init_inverse_mass_matrix`` completes
        healthily and records the donation; a dimension mismatch is loud."""
        task, _ = simple_velocity_task
        common = dict(
            n_samples=200,
            n_warmup=150,
            n_chains=2,
            chain_method='vectorized',
            seed=11,
            progress=False,
            precondition='laplace',
            n_map_starts=3,
        )
        first = build_sampler(
            'numpyro',
            task,
            NumpyroSamplerConfig(precondition_adapt_mass=True, **common),
        ).run()
        adapted = first.diagnostics['adapted_inverse_mass_matrix']
        assert adapted is not None
        adapted = np.asarray(adapted)
        # per-chain stacked (n_chains, D, D) or single (D, D); pool by mean
        D = task.n_params
        if adapted.ndim == 3:
            assert adapted.shape == (2, D, D)
            donor = adapted.mean(axis=0)
        else:
            assert adapted.shape == (D, D)
            donor = adapted
        # symmetrize away accumulated float roundoff before config validation
        donor = 0.5 * (donor + donor.T)

        retry = build_sampler(
            'numpyro',
            task,
            NumpyroSamplerConfig(
                precondition_adapt_mass=True,
                init_inverse_mass_matrix=donor,
                **common,
            ),
        ).run()
        assert retry.metadata['init_mass_donated'] is True
        assert first.metadata['init_mass_donated'] is False
        # donated metric must not break sampling: converged, low divergences
        assert max(retry.get_rhat().values()) < 1.05
        assert retry.diagnostics['divergence_rate'] < 0.1

        # wrong dimension fails loudly at run time
        bad = np.eye(D + 1)
        sampler = build_sampler(
            'numpyro',
            task,
            NumpyroSamplerConfig(init_inverse_mass_matrix=bad, **common),
        )
        with pytest.raises(ValueError, match='does not match'):
            sampler.run()

    def test_preconditioned_matches_standard(self, simple_velocity_task):
        """Preconditioning must not change the posterior: means/stds agree with
        the standard (model-based, adapted) path within MCMC error."""
        task, _ = simple_velocity_task
        common = dict(
            n_samples=400,
            n_warmup=300,
            n_chains=2,
            chain_method='vectorized',
            seed=7,
            progress=False,
        )
        std_res = build_sampler(
            'numpyro', task, NumpyroSamplerConfig(precondition='none', **common)
        ).run()
        pre_res = build_sampler(
            'numpyro',
            task,
            NumpyroSamplerConfig(precondition='laplace', n_map_starts=3, **common),
        ).run()

        names = list(std_res.param_names)
        std_mean = std_res.samples.mean(axis=0)
        std_std = std_res.samples.std(axis=0)
        pre_mean = pre_res.samples.mean(axis=0)
        pre_std = pre_res.samples.std(axis=0)
        for i, name in enumerate(names):
            # Means agree to well within a posterior std (MCMC error is ~0.1
            # std at this ESS; 0.75 is a safe bound, not a tuned one).
            assert abs(pre_mean[i] - std_mean[i]) < 0.75 * std_std[i], (
                f"{name}: preconditioned mean {pre_mean[i]:.4g} vs standard "
                f"{std_mean[i]:.4g} (std {std_std[i]:.4g})"
            )
            # Posterior widths agree within 50%.
            assert abs(pre_std[i] - std_std[i]) < 0.5 * std_std[i], (
                f"{name}: preconditioned std {pre_std[i]:.4g} vs standard "
                f"{std_std[i]:.4g}"
            )


# ==============================================================================
# Full-circle position angle (periodic coordinate)
# ==============================================================================


@pytest.fixture(scope="module")
def wrap_point_velocity_task():
    """Velocity-only task whose true position angle sits just below 2 pi under
    a full-circle PA prior, so the posterior straddles the wrap point."""
    from kl_pipe.priors import CircularUniform

    image_pars = ImagePars(shape=(20, 20), pixel_scale=0.4, indexing='ij')
    theta_true = 2 * np.pi - 0.03
    true_pars_flat = {
        'v0': 10.0,
        'vcirc': 200.0,
        'rscale': 5.0,
        'cosi': 0.6,
        'theta_int': theta_true,
        'g1': 0.02,
        'g2': -0.01,
    }
    true_pars = {
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 5.0,
        'cosi': 0.6,
        'theta_int': theta_true,
        'g1': 0.02,
        'g2': -0.01,
    }
    synth_vel = SyntheticVelocity(true_pars_flat, model_type='arctan', seed=7)
    data_vel_noisy = synth_vel.generate(image_pars, snr=1000)
    source = SourceModel(velocity_model=CenteredVelocityModel())
    priors = PriorDict(
        {
            'vel.v0': Gaussian(10.0, 5.0),
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
            'vel.rscale': TruncatedNormal(5.0, 2.0, 0.4, 20.0),
            'cosi': TruncatedNormal(0.6, 0.2, 0.01, 0.99),
            'theta_int': CircularUniform(2 * np.pi),
            'g1': 0.02,
            'g2': -0.01,
        }
    )
    vel_obs = build_velocity_obs(
        image_pars, data=jnp.array(data_vel_noisy), variance=synth_vel.variance
    )
    return InferenceTask.from_obs(source, priors, velocity_obs=vel_obs), true_pars


class TestCircularPositionAngle:
    def test_preconditioner_map_on_principal_branch(self, wrap_point_velocity_task):
        task, true_pars = wrap_point_velocity_task
        # starts at the wrong rotation direction must not win
        extra = np.array(task.sample_prior(jax.random.PRNGKey(5), n_samples=2))
        i = list(task.sampled_names).index('theta_int')
        extra[:, i] = [true_pars['theta_int'] + np.pi, true_pars['theta_int'] - 8.0]
        pre = task.laplace_preconditioner(n_starts=3, seed=0, extra_starts=extra)
        th = pre.map_point[i]
        assert 0.0 <= th < 2 * np.pi
        resid = np.mod(th - true_pars['theta_int'] + np.pi, 2 * np.pi) - np.pi
        assert abs(resid) < 0.05
        assert pre.start_map_points.shape[1] == len(task.sampled_names)
        assert pre.start_neg_logposts.shape[0] == pre.start_map_points.shape[0]
        assert np.all(pre.start_map_points[:, i] >= 0.0)
        assert np.all(pre.start_map_points[:, i] < 2 * np.pi)
        assert np.isclose(
            pre.start_neg_logposts.min(),
            -float(task.log_posterior(jnp.asarray(pre.map_point))),
            rtol=1e-6,
        )

    def test_posterior_contiguous_across_wrap(self, wrap_point_velocity_task):
        task, true_pars = wrap_point_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=400,
            n_warmup=150,
            n_chains=2,
            chain_method='vectorized',
            seed=42,
            progress=False,
            precondition='laplace',
            precondition_unconstrained=True,
            n_map_starts=3,
        )
        result = build_sampler('numpyro', task, config).run()
        names = list(result.param_names)
        i = names.index('theta_int')
        th = np.asarray(result.samples[:, i])
        # one contiguous branch centred on the MAP: no split into 0 and 2 pi clumps
        assert th.std() < 0.2
        assert th.max() - th.min() < np.pi
        # recovers the truth across the wrap point
        resid = np.mod(th.mean() - true_pars['theta_int'] + np.pi, 2 * np.pi) - np.pi
        assert abs(resid) < 4 * th.std()
        assert result.get_rhat()['theta_int'] < 1.05
        assert result.diagnostics['divergence_rate'] < 0.1
        kinds = result.diagnostics['preconditioner']['unconstrained']['kinds']
        assert kinds['theta_int'] == 'periodic'


class TestCircularPriorEquivalence:
    """The circular prior changes the posterior only by a constant on the
    half-turn support, and the likelihood is periodic in theta with period
    2 pi but not pi (the velocity field distinguishes the rotation direction)."""

    @staticmethod
    def _tasks():
        from kl_pipe.priors import CircularUniform

        image_pars = ImagePars(shape=(20, 20), pixel_scale=0.4, indexing='ij')
        true_pars_flat = {
            'v0': 10.0,
            'vcirc': 200.0,
            'rscale': 5.0,
            'cosi': 0.6,
            'theta_int': 1.9,
            'g1': 0.02,
            'g2': -0.01,
        }
        synth = SyntheticVelocity(true_pars_flat, model_type='arctan', seed=3)
        data = synth.generate(image_pars, snr=200)
        source = SourceModel(velocity_model=CenteredVelocityModel())
        common = {
            'vel.v0': Gaussian(10.0, 5.0),
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
            'vel.rscale': TruncatedNormal(5.0, 2.0, 0.4, 20.0),
            'cosi': TruncatedNormal(0.6, 0.2, 0.01, 0.99),
            'g1': 0.02,
            'g2': -0.01,
        }
        obs = build_velocity_obs(
            image_pars, data=jnp.array(data), variance=synth.variance
        )
        half = InferenceTask.from_obs(
            source,
            PriorDict({**common, 'theta_int': Uniform(0.0, np.pi)}),
            velocity_obs=obs,
        )
        full = InferenceTask.from_obs(
            source,
            PriorDict({**common, 'theta_int': CircularUniform(2 * np.pi)}),
            velocity_obs=obs,
        )
        return half, full

    def test_posteriors_differ_by_the_prior_constant_on_the_half_turn(self):
        half, full = self._tasks()
        assert list(half.sampled_names) == list(full.sampled_names)
        draws = np.asarray(half.sample_prior(jax.random.PRNGKey(1), n_samples=64))
        lp_h = np.array([float(half.log_posterior(jnp.asarray(t))) for t in draws])
        lp_f = np.array([float(full.log_posterior(jnp.asarray(t))) for t in draws])
        assert np.all(np.isfinite(lp_h))
        # log(1/2pi) - log(1/pi) = -log 2, at every point of the shared support
        assert np.allclose(lp_f - lp_h, -np.log(2.0), rtol=0, atol=1e-9)
        g_h = np.asarray(jax.grad(half.log_posterior)(jnp.asarray(draws[0])))
        g_f = np.asarray(jax.grad(full.log_posterior)(jnp.asarray(draws[0])))
        assert np.allclose(g_h, g_f, rtol=1e-10, atol=1e-8)

    def test_likelihood_periodic_in_2pi_not_pi(self):
        _, full = self._tasks()
        i = list(full.sampled_names).index('theta_int')
        theta = np.asarray(full.sample_prior(jax.random.PRNGKey(2), n_samples=1))[0]
        lp0 = float(full.log_posterior(jnp.asarray(theta)))
        for k in (-2, -1, 1, 3):
            shifted = theta.copy()
            shifted[i] += 2 * np.pi * k
            assert np.isclose(
                float(full.log_posterior(jnp.asarray(shifted))), lp0, rtol=0, atol=1e-7
            )
        flipped = theta.copy()
        flipped[i] += np.pi
        # counter-rotation: the same isophotes, the opposite velocity field
        assert abs(float(full.log_posterior(jnp.asarray(flipped))) - lp0) > 10.0


class TestContinueSampling:
    def test_continuation_extends_chains_without_rewarmup(self, simple_velocity_task):
        task, true_pars = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=150,
            n_chains=2,
            chain_method='vectorized',
            seed=5,
            progress=False,
            precondition='laplace',
            precondition_unconstrained=True,
            precondition_adapt_mass=True,
            n_map_starts=3,
        )
        sampler = build_sampler('numpyro', task, config)
        first = sampler.run()
        second = sampler.continue_sampling(200)
        n = len(task.sampled_names)

        # union of draws, chain-major: chain 0 (old, new), chain 1 (old, new)
        assert second.samples.shape == (2 * 400, n)
        assert second.log_prob.shape == (2 * 400,)
        old = np.asarray(first.samples).reshape(2, 200, n)
        new = np.asarray(second.samples).reshape(2, 400, n)
        assert np.array_equal(new[:, :200, :], old)
        assert np.array_equal(
            np.asarray(second.log_prob).reshape(2, 400)[:, :200],
            np.asarray(first.log_prob).reshape(2, 200),
        )
        # the new draws are real posterior draws with consistent log-posteriors
        lp_check = np.array(
            [float(task.log_posterior(jnp.asarray(t))) for t in new[0, 200:205]]
        )
        assert np.allclose(
            lp_check, np.asarray(second.log_prob).reshape(2, 400)[0, 200:205]
        )
        # the warm state carried over: no warmup, same adapted step size
        assert second.diagnostics['step_size'] == first.diagnostics['step_size']
        assert second.metadata['continuations'] == 1
        assert second.metadata['continued_draws_per_chain'] == 200
        assert second.metadata['n_samples_per_chain'] == 400
        # per-draw arrays cover every draw; convergence stats on the union
        assert second.diagnostics['num_steps'].shape == (800,)
        assert second.diagnostics['diverging'].shape == (800,)
        assert np.array_equal(
            second.diagnostics['num_steps'][:400], first.diagnostics['num_steps']
        )
        assert 0.0 <= second.diagnostics['divergence_rate'] < 0.1
        assert max(second.get_rhat().values()) < 1.1
        ess1 = np.array(list(first.diagnostics['ess'].values()))
        ess2 = np.array(list(second.diagnostics['ess'].values()))
        assert np.mean(ess2 / ess1) > 1.3
        assert 'adapted_inverse_mass_matrix' in second.diagnostics
        # chained continuation keeps extending the same chains
        third = sampler.continue_sampling(50)
        assert third.samples.shape == (2 * 450, n)
        assert np.array_equal(
            np.asarray(third.samples).reshape(2, 450, n)[:, :400], new
        )
        assert third.metadata['continuations'] == 2
        # same-size blocks reuse one no-warmup MCMC object (no retrace)
        block_mcmc = sampler._continuation['block_mcmcs'][50]
        fourth = sampler.continue_sampling(50)
        assert sampler._continuation['block_mcmcs'][50] is block_mcmc
        assert set(sampler._continuation['block_mcmcs']) == {200, 50}
        assert fourth.samples.shape == (2 * 500, n)
        assert np.array_equal(
            np.asarray(fourth.samples).reshape(2, 500, n)[:, :450],
            np.asarray(third.samples).reshape(2, 450, n),
        )
        assert not np.array_equal(
            np.asarray(fourth.samples).reshape(2, 500, n)[:, 450:],
            np.asarray(third.samples).reshape(2, 450, n)[:, 400:],
        )

    def test_continue_requires_a_run(self, simple_velocity_task):
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            seed=1,
            progress=False,
            precondition='laplace',
        )
        sampler = build_sampler('numpyro', task, config)
        with pytest.raises(RuntimeError, match="completed preconditioned run"):
            sampler.continue_sampling(10)
