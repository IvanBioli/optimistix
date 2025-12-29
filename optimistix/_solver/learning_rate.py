from typing import cast, ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, Scalar, ScalarLike

from .._custom_types import Y
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


def _typed_asarray(x: ScalarLike) -> Array:
    return jnp.asarray(x)


class _LearningRateState(eqx.Module):
    ls_iter_num: Int[Array, ""]
    accepted: Bool[Array, ""]


class LearningRate(AbstractSearch[Y, FunctionInfo, FunctionInfo, _LearningRateState]):
    """Move downhill by taking a step of the fixed size `learning_rate`."""

    _needs_grad_at_y_eval: ClassVar[bool] = False
    learning_rate: ScalarLike = eqx.field(converter=_typed_asarray)

    def init(self, y: Y, f_info_struct: FunctionInfo) -> _LearningRateState:
        del y, f_info_struct
        return _LearningRateState(
            ls_iter_num=jnp.array(0),
            accepted=jnp.array(False),
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo,
        f_eval_info: FunctionInfo,
        state: _LearningRateState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _LearningRateState]:
        del first_step, y, y_eval, f_info, f_eval_info
        learning_rate = cast(Array, self.learning_rate)
        # LearningRate always accepts, so ls_iter_num is always reset to 0
        new_state = _LearningRateState(
            ls_iter_num=jnp.array(0),
            accepted=jnp.array(True),
        )
        return learning_rate, jnp.array(True), RESULTS.successful, new_state


LearningRate.__init__.__doc__ = """**Arguments:**

- `learning_rate`: The fixed step-size used at each step.
"""
