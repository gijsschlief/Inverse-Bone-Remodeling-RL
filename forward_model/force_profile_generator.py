"""Generates random force profiles for bone remodeling simulations."""

import numpy as np


class ForceProfileGenerator:
    """Generates random force profiles for bone remodeling simulations."""

    def __init__(
        self,
        profile_length: int = 100,
        batch_seed: int | None = None,
    ) -> None:
        """Initialize the force profile generator.

        Args:
        ----
            profile_length (int): Length of the force profile.
            batch_seed (int | None): Seed for random number generation. If None, uses a random seed.

        """
        self._profile_length = profile_length
        self._rng = np.random.default_rng(batch_seed)

    def random(
        self, num_samples: int, force_count_max: int = 5, force_max: float = 10.0
    ) -> np.ndarray:
        """Generate all random force profiles in a fully vectorized way."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        total_elements = profiles.shape[1] * profiles.shape[2]

        # How many non-zero forces per sample?
        counts = self._rng.integers(1, force_count_max, size=num_samples)
        total_forces = np.sum(counts)

        # Flat indices for force assignment
        all_indices = self._rng.choice(
            total_elements,
            size=total_forces,
            replace=True,  # reuse allowed across different samples
        )

        # Random force values
        all_forces = self._rng.uniform(-force_max, self.force_max, size=total_forces)

        # Assign values back to profiles
        flat_profiles = profiles.reshape(num_samples, -1)
        pointer = 0
        for i, count in enumerate(counts):
            if count > 0:
                flat_profiles[i, all_indices[pointer : pointer + count]] = all_forces[
                    pointer : pointer + count
                ]
                pointer += count

        return profiles

    def triangular(self, num_samples: int) -> np.ndarray:
        """Generate triangular force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            # Randomly choose a peak position and height
            peak_position = self._rng.integers(0, self._profile_length)
            peak_height = self._rng.uniform(0, self.force_max)
            side = self._rng.choice([0, 1, 2])

            # Create a triangular profile
            for j in range(self._profile_length):
                if j < peak_position:
                    profiles[i, side, j] = (peak_height / peak_position) * j
                elif j > peak_position:
                    profiles[i, side, j] = (
                        peak_height / (self._profile_length - peak_position)
                    ) * (self._profile_length - j)
                else:
                    profiles[i, side, j] = peak_height
        return profiles

    def square(self, num_samples: int) -> np.ndarray:
        """Generate square force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            start_position = self._rng.integers(0, self._profile_length - 1)
            end_position = self._rng.integers(start_position + 1, self._profile_length)
            height = self._rng.uniform(0, self.force_max)
            side = self._rng.choice([0, 1, 2])

            profiles[i, side, start_position:end_position] = height
        return profiles
