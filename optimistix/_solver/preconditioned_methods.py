"""Preconditioned gradient descent and conjugate gradient methods.

This module provides preconditioned versions of gradient descent and nonlinear CG.
The key innovation is the `AbstractPreconditioner` class, which allows users to define
custom preconditioners that are assembled and applied at each iteration.

Example usage:
```python
import optimistix as optx

# Define a custom preconditioner
class MyPreconditioner(optx.AbstractPreconditioner):
    def init(self, y, f_info_struct):
        return None  # Initial state

    def prepare(self, y, f_info, state, args, options):
        # Assemble the preconditioner based on current iterate
        P = compute_preconditioner(y, f_info)
        return P, state

    def apply(self, preconditioner, grad):
        # Apply preconditioner to gradient
        return preconditioner @ grad

solver = optx.PreconditionedGradientDescent(
    preconditioner=MyPreconditioner(),
    rtol=1e-6,
    atol=1e-6,
)
```
"""

import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    filter_cond,
    lin_to_grad,
    max_norm,
    sum_squares,
    tree_dot,
    tree_full_like,
    tree_where,
)
from .._search import (
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .learning_rate import LearningRate
from .backtracking import BacktrackingArmijo
from .nonlinear_cg import polak_ribiere


# Type variables for preconditioner
Preconditioner = TypeVar("Preconditioner")
PreconditionerState = TypeVar("PreconditionerState")


# Type variable for prepare_aux (data passed from prepare to apply but not stored)
PrepareAux = TypeVar("PrepareAux")


class AbstractPreconditioner(
    eqx.Module, Generic[Y, Preconditioner, PreconditionerState]
):
    """Abstract base class for preconditioners.

    A preconditioner transforms the gradient to improve convergence. At each iteration:
    1. `prepare` is called to assemble/update the preconditioner based on current state
    2. `apply` is called to apply the preconditioner to the gradient

    This separation allows for flexible preconditioner strategies, including:
    - Fixed preconditioners (assembled once)
    - Adaptive preconditioners (updated each iteration)
    - Lazy preconditioners (updated only when certain conditions are met)

    The `prepare_aux` mechanism allows passing non-traceable objects (like class
    instances with closures) from `prepare` to `apply` without storing them in state.
    This is useful for avoiding redundant computations when the same computation is
    needed for both preconditioner construction and application.
    """

    @abc.abstractmethod
    def init(
        self,
        y: Y,
        f_info_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> PreconditionerState:
        """Initialize the preconditioner state.

        **Arguments:**

        - `y`: The initial guess for the optimisation problem.
        - `f_info_struct`: Structure of the function info (for shape/dtype inference).

        **Returns:**

        The initial preconditioner state.
        """

    @abc.abstractmethod
    def prepare(
        self,
        y: Y,
        f_info: FunctionInfo.EvalGrad,
        state: PreconditionerState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Preconditioner, PreconditionerState, Any]:
        """Prepare/assemble the preconditioner for the current iteration.

        This method is called once per iteration, before applying the preconditioner.
        Use this to compute any expensive operations needed for the preconditioner.

        **Arguments:**

        - `y`: The current iterate.
        - `f_info`: Function information including gradient at current iterate.
        - `state`: The current preconditioner state.
        - `args`: Additional arguments passed to the solver.
        - `options`: Solver options.

        **Returns:**

        A tuple of (preconditioner, updated_state, prepare_aux) where:
        - preconditioner: The assembled preconditioner to be used in `apply`.
        - updated_state: The new preconditioner state (stored across iterations).
        - prepare_aux: Optional auxiliary data passed to `apply` but NOT stored
          in state. This is useful for passing non-traceable objects like class
          instances with closures. Can be None if not needed.
        """

    @abc.abstractmethod
    def apply(
        self,
        preconditioner: Preconditioner,
        preconditioner_state: PreconditionerState,
        grad: Y,
        prepare_aux: Any = None,
    ) -> Y:
        """Apply the preconditioner to the gradient.

        **Arguments:**

        - `preconditioner`: The assembled preconditioner from `prepare`.
        - `preconditioner_state`: The current preconditioner state.
        - `grad`: The gradient to precondition.
        - `prepare_aux`: Optional auxiliary data from `prepare`. This is NOT
          stored in state and is only available when `apply` is called immediately
          after `prepare` (i.e., when a step is accepted). When `apply` is called
          from rejected step handling, this will be None.

        **Returns:**

        The preconditioned gradient.
        """


class IdentityPreconditioner(AbstractPreconditioner[Y, None, None]):
    """Identity preconditioner (no preconditioning).

    This is equivalent to standard gradient descent/CG without preconditioning.
    """

    def init(self, y: Y, f_info_struct: PyTree[jax.ShapeDtypeStruct]) -> None:
        return None

    def prepare(
        self,
        y: Y,
        f_info: FunctionInfo.EvalGrad,
        state: None,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[None, None, None]:
        return None, None, None

    def apply(
        self,
        preconditioner: None,
        preconditioner_state: None,
        grad: Y,
        prepare_aux: Any = None,
    ) -> Y:
        return grad


# =============================================================================
# Preconditioned Gradient Descent State
# =============================================================================


class _PreconditionedGDState(
    eqx.Module,
    Generic[Y, Aux, PreconditionerState, Preconditioner],
):
    """State for preconditioned gradient descent."""

    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: Any
    # Updated after each descent step
    f_info: FunctionInfo.EvalPGrad
    aux: Aux
    # Descent direction (preconditioned gradient)
    preconditioned_grad: Y
    # Preconditioner state
    preconditioner_state: PreconditionerState
    current_preconditioner: Preconditioner
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS


class PreconditionedGradientDescent(AbstractMinimiser[Y, Aux, _PreconditionedGDState]):
    """Preconditioned gradient descent with line search.

    At each iteration:
    1. The preconditioner is prepared via `preconditioner.prepare()`
    2. The descent direction is computed as `-P @ grad`
    3. A line search determines the step size

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    search: AbstractSearch[Y, FunctionInfo.EvalPGrad, FunctionInfo.Eval, Any]
    preconditioner: AbstractPreconditioner[Y, Any, Any]

    def __init__(
        self,
        preconditioner: AbstractPreconditioner[Y, Any, Any],
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        search: AbstractSearch[
            Y, FunctionInfo.EvalPGrad, FunctionInfo.Eval, Any
        ] = BacktrackingArmijo(decrease_factor=0.5, slope=0.1),
    ):
        """**Arguments:**

        - `preconditioner`: The preconditioner to use. Must be an instance of
            `AbstractPreconditioner`.
        - `rtol`: Relative tolerance for terminating the solve.
        - `atol`: Absolute tolerance for terminating the solve.
        - `norm`: The norm used to determine convergence.
        - `search`: The (line) search to use at each step. Defaults to
            `BacktrackingArmijo`.
        """
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.search = search
        self.preconditioner = preconditioner

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _PreconditionedGDState:
        f_info = FunctionInfo.EvalPGrad(jnp.zeros(f_struct.shape, f_struct.dtype), y, y)
        f_info_struct = jax.eval_shape(lambda: f_info)
        preconditioner_state = self.preconditioner.init(y, f_info_struct)

        # Get dummy preconditioner for initialization (prepare returns 3 values now)
        dummy_preconditioner, _, _ = jax.eval_shape(
            lambda: self.preconditioner.prepare(
                y, f_info, preconditioner_state, args, options
            )
        )

        return _PreconditionedGDState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            preconditioned_grad=tree_full_like(y, 0),
            preconditioner_state=preconditioner_state,
            current_preconditioner=tree_full_like(
                dummy_preconditioner, 0, allow_static=True
            ),
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _PreconditionedGDState,
        tags: frozenset[object],
    ) -> tuple[Y, _PreconditionedGDState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")

        # Evaluate function at current point
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )

        # Compute gradient if needed by line search
        if self.search._needs_grad_at_y_eval:
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode)
            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad)
        else:
            f_eval_info = FunctionInfo.Eval(f_eval)

        # Line search step
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            f_eval_info,  # pyright: ignore
            state.search_state,
        )

        def accepted(preconditioner_state):
            nonlocal f_eval_info

            # Compute gradient if not already computed
            if not self.search._needs_grad_at_y_eval:
                grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)
            else:
                grad = f_eval_info.grad  # pyright: ignore

            # Prepare and apply preconditioner
            # prepare() now returns (preconditioner, new_state, prepare_aux)
            f_eval_info_for_prepare = FunctionInfo.EvalGrad(f_eval, grad)

            # Add step_info for Trust Region / Levenberg-Marquardt strategies
            # This provides information about the previous step for adaptive mu
            options_with_step_info = {**options}
            options_with_step_info["step_info"] = {
                "f_old": state.f_info.f,
                "f_new": f_eval,
                "step_size": step_size,
                "preconditioned_grad": state.preconditioned_grad,
                "prev_grad": state.f_info.grad,
                "first_step": state.first_step,
            }

            current_preconditioner, new_preconditioner_state, prepare_aux = (
                self.preconditioner.prepare(
                    state.y_eval,
                    f_eval_info_for_prepare,
                    preconditioner_state,
                    args,
                    options_with_step_info,
                )
            )
            # Pass prepare_aux to apply() - this allows reusing computed data
            # (e.g., Gramian) without storing it in state
            preconditioned_grad = self.preconditioner.apply(
                current_preconditioner, new_preconditioner_state, grad, prepare_aux
            )

            # Create EvalPGrad with all gradient info
            f_eval_info_out = FunctionInfo.EvalPGrad(f_eval, grad, preconditioned_grad)

            # Check termination
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)

            return (
                state.y_eval,
                f_eval_info_out,
                aux_eval,
                preconditioned_grad,
                new_preconditioner_state,
                current_preconditioner,
                terminate,
            )

        def rejected(preconditioner_state):
            return (
                y,
                state.f_info,
                state.aux,
                state.preconditioned_grad,
                preconditioner_state,
                state.current_preconditioner,
                jnp.array(False),
            )

        (
            y_new,
            f_info,
            aux,
            preconditioned_grad,
            preconditioner_state,
            current_preconditioner,
            terminate,
        ) = filter_cond(accept, accepted, rejected, state.preconditioner_state)

        # Take step in preconditioned gradient direction
        y_descent = (-step_size * preconditioned_grad**ω).ω
        y_eval = (y_new**ω + y_descent**ω).ω

        result = RESULTS.where(
            search_result == RESULTS.successful, RESULTS.successful, search_result
        )

        new_state = _PreconditionedGDState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            preconditioned_grad=preconditioned_grad,
            preconditioner_state=preconditioner_state,
            current_preconditioner=current_preconditioner,
            terminate=terminate,
            result=result,
        )
        return y_new, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _PreconditionedGDState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _PreconditionedGDState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


# =============================================================================
# Preconditioned Nonlinear CG State
# =============================================================================


class _PreconditionedCGState(
    eqx.Module,
    Generic[Y, Aux, PreconditionerState, Preconditioner],
):
    """State for preconditioned nonlinear CG."""

    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: Any
    # Updated after each descent step
    f_info: FunctionInfo.EvalPGrad
    aux: Aux
    # CG-specific state
    grad: Y
    preconditioned_grad: Y
    search_direction: Y  # The CG search direction
    # Preconditioner state
    preconditioner_state: PreconditionerState
    current_preconditioner: Preconditioner
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS


class PreconditionedNonlinearCG(AbstractMinimiser[Y, Aux, _PreconditionedCGState]):
    """Preconditioned nonlinear conjugate gradient method.

    This uses preconditioned gradients for both the search direction and the
    computation of beta.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    method: Callable[[Y, Y, Y], Scalar]
    search: AbstractSearch[Y, FunctionInfo.EvalPGrad, FunctionInfo.Eval, Any]
    preconditioner: AbstractPreconditioner[Y, Any, Any]

    def __init__(
        self,
        preconditioner: AbstractPreconditioner[Y, Any, Any],
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Scalar] = max_norm,
        method: Callable[[Y, Y, Y], Scalar] = polak_ribiere,
        search: AbstractSearch[
            Y, FunctionInfo.EvalPGrad, FunctionInfo.Eval, Any
        ] = BacktrackingArmijo(decrease_factor=0.5, slope=0.1),
    ):
        """**Arguments:**

        - `preconditioner`: The preconditioner to use. Must be an instance of
            `AbstractPreconditioner`.
        - `rtol`: Relative tolerance for terminating solve.
        - `atol`: Absolute tolerance for terminating solve.
        - `norm`: The norm used to determine convergence.
        - `method`: The function which computes `beta`. Defaults to `polak_ribiere`.
        - `search`: The (line) search to use at each step.
        """
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.method = method
        self.search = search
        self.preconditioner = preconditioner

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _PreconditionedCGState:
        f_info = FunctionInfo.EvalPGrad(jnp.zeros(f_struct.shape, f_struct.dtype), y, y)
        f_info_struct = jax.eval_shape(lambda: f_info)
        preconditioner_state = self.preconditioner.init(y, f_info_struct)

        # Get dummy preconditioner for initialization (prepare returns 3 values now)
        dummy_preconditioner, _, _ = jax.eval_shape(
            lambda: self.preconditioner.prepare(
                y, f_info, preconditioner_state, args, options
            )
        )

        return _PreconditionedCGState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            grad=tree_full_like(y, 0),
            preconditioned_grad=tree_full_like(y, 0),
            search_direction=tree_full_like(y, 0),
            preconditioner_state=preconditioner_state,
            current_preconditioner=tree_full_like(
                dummy_preconditioner, 0, allow_static=True
            ),
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _PreconditionedCGState,
        tags: frozenset[object],
    ) -> tuple[Y, _PreconditionedCGState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")

        # Evaluate function at current point
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )

        # Compute gradient if needed by line search
        if self.search._needs_grad_at_y_eval:
            grad_for_search = lin_to_grad(lin_fn, state.y_eval, autodiff_mode)
            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad_for_search)
        else:
            f_eval_info = FunctionInfo.Eval(f_eval)

        # Line search step
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            f_eval_info,  # pyright: ignore
            state.search_state,
        )

        def accepted(
            preconditioner_state, grad_prev, precond_grad_prev, search_dir_prev
        ):
            # Compute gradient if not already computed
            if not self.search._needs_grad_at_y_eval:
                grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)
            else:
                grad = f_eval_info.grad  # pyright: ignore

            # Prepare and apply preconditioner
            # prepare() now returns (preconditioner, new_state, prepare_aux)
            f_eval_info_for_prepare = FunctionInfo.EvalGrad(f_eval, grad)

            # Add step_info for Trust Region / Levenberg-Marquardt strategies
            options_with_step_info = {**options}
            options_with_step_info["step_info"] = {
                "f_old": state.f_info.f,
                "f_new": f_eval,
                "step_size": step_size,
                "preconditioned_grad": precond_grad_prev,
                "prev_grad": grad_prev,
                "first_step": state.first_step,
            }

            current_preconditioner, new_preconditioner_state, prepare_aux = (
                self.preconditioner.prepare(
                    state.y_eval,
                    f_eval_info_for_prepare,
                    preconditioner_state,
                    args,
                    options_with_step_info,
                )
            )
            # Pass prepare_aux to apply() - this allows reusing computed data
            # (e.g., Gramian) without storing it in state
            preconditioned_grad = self.preconditioner.apply(
                current_preconditioner, new_preconditioner_state, grad, prepare_aux
            )

            # Create EvalPGrad with all gradient info
            f_eval_info_out = FunctionInfo.EvalPGrad(f_eval, grad, preconditioned_grad)

            # Compute beta using preconditioned gradients (Polak-Ribière style)
            beta = self.method(
                grad,
                grad_prev,
                search_dir_prev,
                preconditioned_grad=preconditioned_grad,
                preconditioned_grad_prev=precond_grad_prev,
            )

            # Compute new search direction: -P@g + beta * d_prev
            neg_precond_grad = (-(preconditioned_grad**ω)).ω
            new_search_dir = (neg_precond_grad**ω + beta * search_dir_prev**ω).ω

            # Check if this is a descent direction. Use preconditioned gradient if not.
            new_search_dir = tree_where(
                tree_dot(grad, new_search_dir) < 0,
                new_search_dir,
                neg_precond_grad,
            )

            # Check termination
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)

            return (
                state.y_eval,
                f_eval_info_out,
                aux_eval,
                grad,
                preconditioned_grad,
                new_search_dir,
                new_preconditioner_state,
                current_preconditioner,
                terminate,
            )

        def rejected(
            preconditioner_state, grad_prev, precond_grad_prev, search_dir_prev
        ):
            return (
                y,
                state.f_info,
                state.aux,
                grad_prev,
                precond_grad_prev,
                search_dir_prev,
                preconditioner_state,
                state.current_preconditioner,
                jnp.array(False),
            )

        (
            y_new,
            f_info,
            aux,
            grad,
            preconditioned_grad,
            search_direction,
            preconditioner_state,
            current_preconditioner,
            terminate,
        ) = filter_cond(
            accept,
            accepted,
            rejected,
            state.preconditioner_state,
            state.grad,
            state.preconditioned_grad,
            state.search_direction,
        )

        # Take step in search direction
        y_descent = (step_size * search_direction**ω).ω
        y_eval = (y_new**ω + y_descent**ω).ω

        result = RESULTS.where(
            search_result == RESULTS.successful, RESULTS.successful, search_result
        )

        new_state = _PreconditionedCGState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            grad=grad,
            preconditioned_grad=preconditioned_grad,
            search_direction=search_direction,
            preconditioner_state=preconditioner_state,
            current_preconditioner=current_preconditioner,
            terminate=terminate,
            result=result,
        )
        return y_new, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _PreconditionedCGState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _PreconditionedCGState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}
