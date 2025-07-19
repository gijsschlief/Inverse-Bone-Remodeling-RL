"""Callback to render the environment at specified intervals."""

from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):
    """Callback to render the environment at specified intervals."""

    def __init__(self, render_freq: int = 10, verbose: int = 0) -> None:
        """Initialize the render callback.

        Args:
        ----
            render_freq (int): Frequency of rendering in terms of steps.
            verbose (int): Verbosity level.

        """
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.get_attr("render")[0]()
        return True
