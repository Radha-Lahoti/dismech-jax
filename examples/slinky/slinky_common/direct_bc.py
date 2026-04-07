"""
Build :class:`dismech_jax.BatchedDirectBC` from dataset objects loaded from NPZ.

Several experiment notebooks store prescribed displacements in ``xb`` with shape
``(n_traj, n_lambda, n_bc)`` and share the same helper to wrap them for ``rod.solve``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class _DirectBCDatasetView(Protocol):
    """Minimal surface required to build a batched direct BC."""

    idx_b: Any
    xb: Any
    qs: Any
    lambdas: Any


class DirectBCFactory:
    """
    Constructs ``BatchedDirectBC`` instances from slinky test-case containers.

    Keeping this as a small class (instead of only a bare function) makes it easy
    to swap the backend module in tests or to attach logging later.
    """

    def __init__(self, djx: Any | None = None) -> None:
        # Lazy default avoids import cost for notebooks that only need typing.
        self._djx = djx

    @property
    def djx(self) -> Any:
        if self._djx is None:
            import dismech_jax as djx

            self._djx = djx
        return self._djx

    def from_dataset(
        self,
        dataset: _DirectBCDatasetView,
        trajectory_index: int | None = None,
        *,
        verbose: bool = False,
    ) -> Any:
        """
        Parameters
        ----------
        dataset
            Object with ``idx_b``, ``xb``, ``qs``, and optional ``lambdas``.
        trajectory_index
            If ``None``, use the full batch (``xb`` may be 3D). If an ``int``,
            select trajectory ``i`` (``xb[i]`` and possibly ``idx_b[i]``).
        verbose
            When ``True``, print ``xb`` shape (matches older notebook debugging).
        """
        lambdas = (
            dataset.lambdas
            if dataset.lambdas is not None
            else jnp.linspace(0.0, 1.0, dataset.qs.shape[1])
        )

        if trajectory_index is None:
            idx_b = dataset.idx_b
            xb = dataset.xb
        else:
            i = trajectory_index
            idx_b = dataset.idx_b if dataset.idx_b.ndim == 1 else dataset.idx_b[i]
            xb = dataset.xb[i]

        if verbose:
            print("xb.shape", getattr(xb, "shape", None))

        return self.djx.BatchedDirectBC(
            idx_b=idx_b,
            xb=xb,
            lambdas=lambdas,
        )


_default_factory = DirectBCFactory()


def make_bc_from_testcase(
    dataset: _DirectBCDatasetView,
    i: int | None = None,
    *,
    djx: Any | None = None,
    verbose: bool = True,
) -> Any:
    """
    Notebook-compatible alias: ``make_bc_from_testcase(valid)``.

    Pass ``djx=dismech_jax`` if you import the library under another name.
    """
    factory = DirectBCFactory(djx=djx) if djx is not None else _default_factory
    return factory.from_dataset(dataset, trajectory_index=i, verbose=verbose)
