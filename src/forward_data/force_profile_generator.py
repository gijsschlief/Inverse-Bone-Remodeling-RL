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

    def impulse(
        self,
        num_samples: int,
        force_count_max: int = 30,
        force_max: float = 10.0,
    ) -> np.ndarray:
        """Generate all random force profiles in a fully vectorized way."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        total_elements = profiles.shape[1] * profiles.shape[2]

        force_selection_probabilities = np.exp(-np.arange(1, force_count_max + 1))
        force_selection_probabilities /= force_selection_probabilities.sum()
        counts = self._rng.choice(
            np.arange(1, force_count_max + 1),
            size=num_samples,
            p=force_selection_probabilities,
        )
        total_forces = np.sum(counts)

        # Flat indices for force assignment
        all_indices = self._rng.choice(
            total_elements,
            size=total_forces,
            replace=True,  # reuse allowed across different samples
        )

        # Random force values
        all_forces = self._rng.uniform(-force_max, force_max, size=total_forces)

        # Assign values back to profiles
        flat_profiles = profiles.reshape(num_samples, -1)
        pointer = 0
        for i, count in enumerate(counts):
            if count > 0:
                flat_profiles[i, all_indices[pointer : pointer + count]] = all_forces[
                    pointer : pointer + count
                ]
                pointer += int(count)

        return profiles

    def triangular(self, num_samples: int, force_max: float = 10.0) -> np.ndarray:
        """Generate triangular force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            # Randomly choose a peak position and height
            peak_position = self._rng.integers(0, self._profile_length - 1)
            peak_height = self._rng.uniform(-force_max, force_max)
            side = self._rng.choice([0, 1, 2])

            # Create a triangular profile
            for j in range(self._profile_length):
                if j < peak_position:
                    profiles[i, side, j] = (peak_height / peak_position) * j
                elif j > peak_position:
                    profiles[i, side, j] = (
                        peak_height / (self._profile_length - 1 - peak_position)
                    ) * (self._profile_length - 1 - j)
                else:
                    profiles[i, side, j] = peak_height
        return profiles

    def square(self, num_samples: int, force_max: float = 10.0) -> np.ndarray:
        """Generate square force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            start_position = self._rng.integers(0, self._profile_length - 1)
            end_position = self._rng.integers(start_position + 1, self._profile_length)
            height = self._rng.uniform(0, force_max)
            side = self._rng.choice([0, 1, 2])

            profiles[i, side, start_position:end_position] = height
        return profiles

    def gaussian(self, num_samples: int, force_max: float = 10.0) -> np.ndarray:
        """Generate Gaussian force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            mean_position = self._rng.integers(0, self._profile_length)
            std_dev = self._rng.uniform(1, self._profile_length / 10)
            height = self._rng.uniform(0, force_max)
            side = self._rng.choice([0, 1, 2])

            x = np.arange(self._profile_length)
            gaussian_profile = height * np.exp(
                -((x - mean_position) ** 2) / (2 * std_dev**2),
            )
            profiles[i, side, :] = gaussian_profile
        return profiles

    def ramp(self, num_samples: int, force_max: float = 10.0) -> np.ndarray:
        """Generate ramp force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            start_position = self._rng.integers(0, self._profile_length - 1)
            end_position = self._rng.integers(start_position + 1, self._profile_length)
            height = self._rng.uniform(0, force_max)
            side = self._rng.choice([0, 1, 2])

            for j in range(self._profile_length):
                if j < start_position:
                    profiles[i, side, j] = 0
                elif j < end_position:
                    profiles[i, side, j] = (
                        height / (end_position - start_position)
                    ) * (j - start_position)
                else:
                    profiles[i, side, j] = height
        return profiles

    def merger(self, num_samples: int, force_max: float = 10.0) -> np.ndarray:
        """Generate merged force profiles from different shapes."""
        profile_count = self._rng.choice(
            [1, 2, 3, 4, 5],
            size=num_samples,
            p=[0.8, 0.13, 0.05, 0.01, 0.01],
        )
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            for _ in range(profile_count[i]):
                shape_type = self._rng.choice(
                    ["triangular", "square", "gaussian", "impulse", "ramp"],
                    p=[0.2, 0.2, 0.2, 0.2, 0.2],
                )
                if shape_type == "triangular":
                    profiles[i] += self.triangular(1, force_max)[0]
                elif shape_type == "square":
                    profiles[i] += self.square(1, force_max)[0]
                elif shape_type == "gaussian":
                    profiles[i] += self.gaussian(1, force_max)[0]
                elif shape_type == "impulse":
                    profiles[i] += self.impulse(1, force_max=force_max)[0]
                elif shape_type == "ramp":
                    profiles[i] += self.ramp(1, force_max)[0]

            # Compute energy (L2 norm squared) and apply random scaling
            target_energy = self._rng.normal(
                loc=force_max**2 / 2,
                scale=force_max**2 / 4,
            )
            target_energy = max(target_energy, 1e-6)
            actual_energy = np.sum(profiles[i] ** 2)
            eps = 1e-6
            if actual_energy > 0:
                scaling_factor = np.sqrt(target_energy / (actual_energy + eps))
                profiles[i] *= scaling_factor * self._rng.uniform(0.5, 2)
        return profiles
