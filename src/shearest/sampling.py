"""MCLMC initialisation and adaptation.

Centralises the MCLMC setup that used to live inline in ``scripts/run.py``:

- Gradient diagnostics at MAP (pre/post-MAP reduction, per-parameter-group
  ‖∇‖) — useful to detect MAPs that didn't converge.
- Gradient-aware initial ``(L, step_size)`` with a ``lr_map`` floor.
- Diagonal inverse-mass-matrix construction from CLI args.
- Phase 1+2 + phase 3 adaptation with a retry loop and a ``chain_ok`` gate
  that skips phase 3 when phases 1+2 collapse.
- Diagnostic logging of the adapted mass matrix and a save to disk.

The public entry point is :func:`setup_mclmc`. The five smaller helpers are
exported as well so they can be unit-tested individually.

The MCLMC hyper-parameters that aren't (yet) on the CLI live at the top of
this module as named constants — discoverable and easy to tune in one
place. If you want to expose any of them through ``--mclmc_*`` flags later,
the constants just become defaults.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import blackjax
import blackjax.adaptation.mclmc_adaptation as mclmc_adj
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


# --------------------------------------------------------------------------- #
# MCLMC tuning hyper-parameters                                               #
# --------------------------------------------------------------------------- #
#: Target per-dimension energy variance during phase-1+2 adaptation. Smaller
#: values demand a tighter step_size and pay in warmup cost.
DESIRED_ENERGY_VAR: float = 1e-3

#: How much weight to put on the predictor's step_size estimate vs. the
#: current value during phase-1+2 EMA updates.
TRUST_IN_ESTIMATE: float = 2.0

#: Effective sample count used in the variance estimator that drives phase 2.
NUM_EFFECTIVE_SAMPLES: int = 50

#: Fractions of ``n_warmup`` allocated to each adaptation phase.
#: Together they sum to 1.0 — the full warmup budget is used.
FRAC_TUNE1: float = 0.4
FRAC_TUNE2: float = 0.4
FRAC_TUNE3: float = 0.2

#: ``L = Lfactor · step_size · mean(num_steps/ess)`` during phase 3 (ESS-based L).
L_FACTOR_PHASE3: float = 0.4

#: How many times to retry adaptation when phases 1+2 collapse step_size or L to ~0.
MAX_ADAPT_ATTEMPTS: int = 10

#: Threshold below which we consider step_size or L to have collapsed (dead chain).
COLLAPSE_THRESHOLD: float = 1e-10


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def compute_map_gradient_diagnostics(
    log_prob_fn,
    init_val_map,
    init_val_prior,
    logger: Optional[logging.Logger] = None,
):
    """Compute ``ndim``, the per-parameter-group ``‖∇‖`` at MAP, and the
    pre→post-MAP gradient-norm reduction. Returns
    ``(ndim, first_chain_init, grad_norm)``.

    Logs:
    - The overall pre/post-MAP reduction factor.
    - For each parameter group, ``‖∇‖`` and ``max|∂|``.

    A small reduction factor signals that the MAP didn't converge well — the
    downstream gradient-aware ``step_size`` formula will then underestimate.
    """
    first_chain_init = jax.tree.map(lambda x: x[0], init_val_map)
    ndim = sum(v.size for v in jax.tree.leaves(first_chain_init))
    grad_at_map = jax.grad(log_prob_fn)(first_chain_init)
    grad_norm = float(jnp.linalg.norm(ravel_pytree(grad_at_map)[0]))

    first_chain_pre = jax.tree.map(lambda x: x[0], init_val_prior)
    grad_pre_norm = float(
        jnp.linalg.norm(ravel_pytree(jax.grad(log_prob_fn)(first_chain_pre))[0])
    )
    reduction = grad_pre_norm / grad_norm if grad_norm > 0 else float("inf")

    if logger is not None:
        logger.info(
            f"MAP gradient reduction: ‖∇‖ pre-MAP={grad_pre_norm:.3e}, "
            f"post-MAP={grad_norm:.3e}, factor={reduction:.1f}x"
        )
        logger.info("Per-group ‖∇‖ at MAP:")
        for k in sorted(grad_at_map.keys()):
            g_flat = grad_at_map[k].ravel()
            logger.info(
                f"  {k:>12s}: ‖∇‖={float(jnp.linalg.norm(g_flat)):.3e}, "
                f"max|∂|={float(jnp.max(jnp.abs(g_flat))):.3e}"
            )

    return ndim, first_chain_init, grad_norm


def compute_initial_step_size_and_L(
    ndim: int,
    grad_norm: float,
    args,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float]:
    """Gradient-aware initial step_size, floored at ``args.lr_map``.

    The formula keeps the leapfrog energy error in a safe regime:
    overshooting causes a NaN cascade that collapses ``step_size`` to 0
    permanently. Undershooting only costs warmup steps.

    ``L = √dim · step_size`` (coupled momentum-decoherence rate).
    """
    formula = float(jnp.sqrt(ndim) / grad_norm) * (DESIRED_ENERGY_VAR / 1e-2) ** 0.25
    if formula < args.lr_map:
        initial_step_size = args.lr_map
        source = f"floor=lr_map ({args.lr_map:.3e}; formula gave {formula:.3e})"
    else:
        initial_step_size = formula
        source = f"gradient formula ({formula:.3e})"
    initial_L = float(jnp.sqrt(ndim)) * initial_step_size

    if logger is not None:
        logger.info(
            f"MCLMC init: ndim={ndim}, grad_norm={grad_norm:.3e}, "
            f"initial_step_size={initial_step_size:.3e}, "
            f"initial_L={initial_L:.3e} [{source}]"
        )

    return initial_step_size, initial_L


def build_inverse_mass_matrix(
    args,
    first_chain_init,
    ndim: int,
    logger: Optional[logging.Logger] = None,
):
    """Build the initial diagonal inverse mass matrix from CLI args.

    Dispatch order (first match wins):
    1. ``--mclmc_inv_mass_file`` → load the diagonal from ``.npy``.
    2. ``--mclmc_inv_mass_shear`` → diagonal with that value on the g1/g2
       slots, 1.0 elsewhere (per-parameter-group, sorted-key order).
    3. neither → ``jnp.ones((ndim,))``.

    Note: this is only the **initial** mass — phase 2 of adaptation refits a
    diagonal mass from the running sample covariance unless adaptation is
    skipped via ``--mclmc_L`` + ``--step_size``.
    """
    if args.mclmc_inv_mass_file is not None:
        inverse_mass_matrix = jnp.array(np.load(args.mclmc_inv_mass_file))
        if logger is not None:
            logger.info(
                f"MCLMC inverse mass matrix loaded from {args.mclmc_inv_mass_file}"
            )
            logger.info(
                f"  shape={inverse_mass_matrix.shape}, "
                f"min={inverse_mass_matrix.min():.6f}, "
                f"max={inverse_mass_matrix.max():.6f}, "
                f"ratio={inverse_mass_matrix.max()/inverse_mass_matrix.min():.1f}"
            )
        return inverse_mass_matrix

    if args.mclmc_inv_mass_shear is not None:
        parts = []
        for k in sorted(first_chain_init.keys()):
            val = args.mclmc_inv_mass_shear if k in ("g1", "g2") else 1.0
            parts.append(jnp.full(first_chain_init[k].size, val))
        inverse_mass_matrix = jnp.concatenate(parts)
        if logger is not None:
            logger.info(
                f"MCLMC diagonal inverse mass matrix: "
                f"g1/g2={args.mclmc_inv_mass_shear}, others=1.0"
            )
        return inverse_mass_matrix

    return jnp.ones((ndim,))


def run_mclmc_adaptation(
    log_prob_fn,
    first_chain_state,
    initial_params: mclmc_adj.MCLMCAdaptationState,
    n_warmup: int,
    key_tune,
    logger: Optional[logging.Logger] = None,
) -> mclmc_adj.MCLMCAdaptationState:
    """Run phases 1+2 + phase 3 with a retry loop. Returns the final
    :class:`MCLMCAdaptationState`. Raises :class:`RuntimeError` if every
    attempt collapses ``L`` or ``step_size`` to ~0.

    Phase 1+2 (``make_L_step_size_adaptation``) jointly adapts ``step_size``,
    ``L`` and the diagonal mass matrix. Phase 3 (``make_adaptation_L``)
    refines ``L`` from an ESS estimate — but only if phase 1+2 didn't
    collapse, because phase 3 on a dead chain would just propagate the
    collapse.
    """

    def mclmc_factory(inverse_mass_matrix):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=log_prob_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )

    ndim = initial_params.inverse_mass_matrix.size
    L_step_size_adapt = mclmc_adj.make_L_step_size_adaptation(
        kernel=mclmc_factory,
        dim=ndim,
        frac_tune1=FRAC_TUNE1,
        frac_tune2=FRAC_TUNE2,
        desired_energy_var=DESIRED_ENERGY_VAR,
        trust_in_estimate=TRUST_IN_ESTIMATE,
        num_effective_samples=NUM_EFFECTIVE_SAMPLES,
        diagonal_preconditioning=True,
    )

    parameters = initial_params
    for attempt in range(1, MAX_ADAPT_ATTEMPTS + 1):
        if logger is not None:
            logger.info(
                f"MCLMC adaptation attempt {attempt}/{MAX_ADAPT_ATTEMPTS}..."
            )
        key_tune, key_retry = jax.random.split(key_tune)
        key_phase12, key_phase3 = jax.random.split(key_retry)

        # Phase 1+2: adapt step_size, L, and inverse_mass_matrix together.
        adapted_state, parameters = L_step_size_adapt(
            first_chain_state, initial_params, n_warmup, key_phase12
        )
        if logger is not None:
            logger.info(
                f"  After phase 1+2: L={float(parameters.L):.6f}, "
                f"step_size={float(parameters.step_size):.8f}"
            )

        # Phase 3: refine L via ESS. Skip on collapsed chain.
        chain_ok = (
            float(parameters.L) > COLLAPSE_THRESHOLD
            and float(parameters.step_size) > COLLAPSE_THRESHOLD
        )
        if chain_ok and FRAC_TUNE3 > 0:
            adapted_kernel = mclmc_factory(parameters.inverse_mass_matrix)
            adapted_state, parameters = mclmc_adj.make_adaptation_L(
                adapted_kernel, frac=FRAC_TUNE3, Lfactor=L_FACTOR_PHASE3
            )(adapted_state, parameters, n_warmup, key_phase3)
            if logger is not None:
                logger.info(
                    f"  After phase 3:   L={float(parameters.L):.6f}, "
                    f"step_size={float(parameters.step_size):.8f}"
                )
        elif not chain_ok and logger is not None:
            logger.info("  Phase 1+2 collapsed; skipping phase 3")

        if parameters.step_size > 0 and parameters.L > 0:
            return parameters

        if logger is not None:
            logger.info(
                f"Adaptation failed (step_size={parameters.step_size}, "
                f"L={parameters.L}), retrying..."
            )

    msg = (
        f"MCLMC adaptation failed after {MAX_ADAPT_ATTEMPTS} attempts: "
        f"step_size={parameters.step_size}, L={parameters.L}"
    )
    if logger is not None:
        logger.info(msg)
    raise RuntimeError(msg)


def save_inverse_mass_diagnostics(
    parameters: mclmc_adj.MCLMCAdaptationState,
    first_chain_init,
    out_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log per-group inv-mass stats and save the full vector to ``out_dir``.

    For scalar masses, just logs the value. For diagonal masses, logs
    ``min/max/median`` overall and per parameter group, then writes
    ``mclmc_inv_mass_matrix.npy``.
    """
    inv = parameters.inverse_mass_matrix
    if not (hasattr(inv, "shape") and inv.ndim > 0):
        if logger is not None:
            logger.info(f"Inverse mass matrix: scalar = {inv}")
        return

    if logger is not None:
        logger.info(
            f"Inverse mass matrix: min={inv.min():.6f}, max={inv.max():.6f}, "
            f"median={jnp.median(inv):.6f}, ratio={inv.max()/inv.min():.1f}"
        )

    offset = 0
    for k in sorted(first_chain_init.keys()):
        size = first_chain_init[k].size
        chunk = inv[offset : offset + size]
        if logger is not None:
            logger.info(
                f"  {k:>6s} [{size:4d}]: min={chunk.min():.6f}, "
                f"max={chunk.max():.6f}, median={jnp.median(chunk):.6f}"
            )
        offset += size

    np.save(os.path.join(out_dir, "mclmc_inv_mass_matrix.npy"), np.array(inv))
    if logger is not None:
        logger.info(
            f"Saved inverse mass matrix to {out_dir}/mclmc_inv_mass_matrix.npy"
        )


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def setup_mclmc(
    log_prob_fn,
    init_val_map,
    init_val_prior,
    key_warmup,
    args,
    out_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[mclmc_adj.MCLMCAdaptationState, jax.Array]:
    """End-to-end MCLMC setup: diagnostics → initial step_size/L → inverse
    mass matrix → adaptation (or CLI override) → diagnostics + save.

    Parameters
    ----------
    log_prob_fn
        Jitted joint log-density, conditioned on data.
    init_val_map
        Per-chain MAP positions (output of ``find_map``).
    init_val_prior
        Per-chain prior-draw positions (only used to log the pre/post-MAP
        gradient reduction).
    key_warmup
        PRNG key reserved for warmup. Will be split internally into chain-init
        and adaptation keys.
    args
        The parsed CLI namespace.
    out_dir
        Directory where the inverse-mass matrix is saved.
    logger
        Logger for diagnostics (``None`` to suppress).

    Returns
    -------
    parameters : MCLMCAdaptationState
        Adapted (or CLI-overridden) ``(L, step_size, inverse_mass_matrix)``.
        Pass to ``blackjax.mclmc(log_prob_fn, **parameters._asdict())``.
    key_init_chains : jax.Array
        Per-chain init keys for ``jax.vmap(kernel.init)(init_val, key_init_chains)``.
    """
    key_init, key_tune = jax.random.split(key_warmup)
    key_init_chains = jax.random.split(key_init, args.num_chains)

    # 1) Gradient diagnostics at MAP + initial step_size / L.
    ndim, first_chain_init, grad_norm = compute_map_gradient_diagnostics(
        log_prob_fn, init_val_map, init_val_prior, logger
    )
    initial_step_size, initial_L = compute_initial_step_size_and_L(
        ndim, grad_norm, args, logger
    )

    # 2) Initial diagonal inverse mass matrix from CLI args.
    inverse_mass_matrix = build_inverse_mass_matrix(
        args, first_chain_init, ndim, logger
    )

    # 3) Build the initial integrator state for adaptation.
    temp_kernel = blackjax.mclmc(
        log_prob_fn,
        step_size=initial_step_size,
        L=initial_L,
        inverse_mass_matrix=inverse_mass_matrix,
    )
    first_chain_state = temp_kernel.init(first_chain_init, key_init_chains[0])

    # 4) Adaptation — skipped when CLI provides both --mclmc_L and --step_size.
    if args.mclmc_L is not None and args.step_size is not None:
        if logger is not None:
            logger.info(
                f"Skipping MCLMC adaptation: using L={args.mclmc_L}, "
                f"step_size={args.step_size}"
            )
        parameters = mclmc_adj.MCLMCAdaptationState(
            L=jnp.array(args.mclmc_L),
            step_size=jnp.array(args.step_size),
            inverse_mass_matrix=jnp.array(inverse_mass_matrix),
        )
        return parameters, key_init_chains

    initial_params = mclmc_adj.MCLMCAdaptationState(
        L=jnp.array(initial_L),
        step_size=jnp.array(initial_step_size),
        inverse_mass_matrix=jnp.array(inverse_mass_matrix),
    )
    parameters = run_mclmc_adaptation(
        log_prob_fn,
        first_chain_state,
        initial_params,
        args.n_warmup,
        key_tune,
        logger,
    )

    if logger is not None:
        logger.info(f"Step size: {parameters.step_size}")
        logger.info(f"L: {parameters.L}")
    save_inverse_mass_diagnostics(parameters, first_chain_init, out_dir, logger)

    return parameters, key_init_chains
