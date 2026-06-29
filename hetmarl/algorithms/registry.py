"""Registry mapping ``algorithm_name`` to the classes/flags that vary by algorithm.

Centralizes the ``if algorithm_name == "macmat"/"amat"`` chains that were spread
across the runner and the policy. Class resolvers are lazy (imported on demand)
to avoid import cycles and to keep optional dependencies out of a default run.

Frontier-search baselines (``ft_*``) do not use a learned policy; callers detect
them with :func:`is_frontier_algorithm` and skip policy/trainer construction.
"""
from dataclasses import dataclass
from typing import Callable


def _transformer_policy():
    from hetmarl.algorithms.transformer_policy import TransformerPolicy
    return TransformerPolicy


def _macmat_trainer():
    from hetmarl.algorithms.macmat_trainer import MACMAT_Trainer
    return MACMAT_Trainer


def _macmat_transformer():
    from hetmarl.algorithms.macmat.algorithm.macmat_transformer import (
        AsynchronousClassBasedLiquidMultiAgentTransformer,
    )
    return AsynchronousClassBasedLiquidMultiAgentTransformer


def _amat_transformer():
    from hetmarl.algorithms.amat.algorithm.amat_transformer import (
        AsynchronousMultiAgentTransformer,
    )
    return AsynchronousMultiAgentTransformer


@dataclass(frozen=True)
class AlgorithmSpec:
    """Per-algorithm wiring.

    Attributes are zero-arg callables returning the class (lazy import). The
    ``class_conditioned`` flag records whether the transformer takes MACMAT's
    class-based-action / graph-attention arguments.
    """
    policy: Callable
    trainer: Callable
    transformer: Callable
    class_conditioned: bool


ALGORITHMS = {
    "macmat": AlgorithmSpec(_transformer_policy, _macmat_trainer, _macmat_transformer, class_conditioned=True),
    # AMAT reuses TransformerPolicy + the MACMAT trainer (the dedicated amat_trainer
    # is currently unused); only the transformer core differs.
    "amat": AlgorithmSpec(_transformer_policy, _macmat_trainer, _amat_transformer, class_conditioned=False),
}


def is_frontier_algorithm(name: str) -> bool:
    """Whether ``name`` is a frontier-search baseline (``ft_*``) with no learned policy."""
    return "ft" in name


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    """Return the :class:`AlgorithmSpec` for ``name`` or raise ``NotImplementedError``."""
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm_name: {name!r}")
