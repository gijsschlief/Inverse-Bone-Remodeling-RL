"""Callback to render the environment at specified intervals."""

from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):
    """Callback to render the environment at specified intervals."""

    def __init__(self, render_freq: int = 10, environment_index: int = 0, verbose: int = 0) -> None:
        """Initialize the render callback.

        Args:
        ----
            render_freq (int): Frequency of rendering in terms of steps.
            environment_index (int): Index of the environment to render.
            verbose (int): Verbosity level.

        """
        super().__init__(verbose)
        self.render_freq = render_freq
        self.environment_index = environment_index

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            return True

        vectorized_environment = self.training_env
        try:
            sub_environment = vectorized_environment.envs[self.environment_index]
        except AttributeError:
            sub_environment = vectorized_environment

        sub_environment.render(mode="human")
        return False #ignore plotting for now: True
