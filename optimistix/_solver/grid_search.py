from typing import cast, ClassVar, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar

from .._custom_types import Y
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


class _GridSearchState(eqx.Module):
    index: Int[Array, ""]  # Current step size index being evaluated
    best_index: Int[Array, ""]  # Index of the best step size found so far
    best_f: Scalar  # Best function value found so far
    restoring_best: Bool[Array, ""]  # Whether we're in the extra step to restore best
    ls_iter_num: Int[Array, ""]  # Number of line search steps since last accept
    accepted: Bool[Array, ""]  # Whether the last step was accepted


_FnInfo: TypeAlias = FunctionInfo
_FnEvalInfo: TypeAlias = FunctionInfo


class GridSearch(AbstractSearch[Y, _FnInfo, _FnEvalInfo, _GridSearchState]):
    """Perform a grid search over a predefined set of step sizes.

    At each optimization step, this search evaluates the objective function at each
    step size in the grid, and selects the step size that gives the best (lowest)
    function value.
    """

    n_steps: Int
    step_sizes: Float[Array, " n_steps"]
    _needs_grad_at_y_eval: ClassVar[bool] = False

    def __init__(self, step_sizes: Float[Array, " n_steps"]):
        """**Arguments:**

        - `step_sizes`: A 1D array of step sizes to try. The search will evaluate
            each step size and select the one that minimizes the objective function.
        """
        self.step_sizes = jnp.asarray(step_sizes)
        assert self.step_sizes.ndim == 1
        self.n_steps = self.step_sizes.shape[0]

    def init(self, y: Y, f_info_struct: _FnInfo) -> _GridSearchState:
        del y, f_info_struct
        return _GridSearchState(
            index=jnp.array(-1, dtype=jnp.int32),
            best_index=jnp.array(-1, dtype=jnp.int32),
            best_f=jnp.array(jnp.inf),
            restoring_best=jnp.array(False),
            ls_iter_num=jnp.array(0),
            accepted=jnp.array(False),
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: _GridSearchState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _GridSearchState]:
        del y, y_eval  # Not needed

        f_eval = f_eval_info.as_min()

        # If we're restoring the best step size, accept now
        # (y_eval was computed with the best step size from previous iteration)
        accept_restore = state.restoring_best

        # Update best if current is better (only when not restoring)
        is_better = (f_eval < state.best_f) & ~state.restoring_best
        best_f = jnp.where(is_better, f_eval, state.best_f)
        best_index = jnp.where(is_better, state.index, state.best_index)

        # Move to next index (only when not restoring)
        next_index = jnp.where(state.restoring_best, state.index, state.index + 1)

        # Have we evaluated all step sizes?
        finished_grid = next_index >= self.n_steps

        # Accept on first_step or when restoring best
        accept = first_step | accept_restore

        # Determine the step size to use next
        # If finished_grid: propose the best step size (will be evaluated in next iter)
        # If restoring: this doesn't matter as we're accepting
        # Otherwise: use the next step size in the grid
        step_size = jnp.where(
            finished_grid | state.restoring_best,
            self.step_sizes[best_index],
            self.step_sizes[next_index % self.n_steps],  # Modulo to avoid out-of-bounds
        )
        step_size = cast(Scalar, step_size)

        # Update state
        # ls_iter_num: reset to 0 when accepted, otherwise increment
        ls_iter_num = jnp.where(accept, jnp.array(0), state.ls_iter_num + 1)

        # When finished_grid and not restoring: enter restoring mode
        # When restoring: reset everything for next iteration
        new_restoring_best = finished_grid & ~state.restoring_best
        new_index = jnp.where(
            accept,
            jnp.array(0, dtype=jnp.int32),
            next_index,
        )
        new_best_index = jnp.where(
            accept,
            jnp.array(0, dtype=jnp.int32),
            best_index,
        )
        new_best_f = jnp.where(
            accept,
            jnp.array(jnp.inf),
            best_f,
        )

        new_state = _GridSearchState(
            index=new_index,
            best_index=new_best_index,
            best_f=new_best_f,
            restoring_best=new_restoring_best,
            ls_iter_num=ls_iter_num,
            accepted=accept,
        )

        return step_size, accept, RESULTS.successful, new_state
