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
        force_min: float = 0.1,
    ) -> np.ndarray:
        """Generate all random force profiles in a fully vectorized way."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        total_elements = profiles.shape[1] * profiles.shape[2]

        # Determine number of forces per sample using lognormal distribution
        profile_count = self._log_normal_sampling(mean=0.5, sigma=0.8, size=num_samples, rng=self._rng).astype(int)
        counts = np.clip(profile_count, 1, force_count_max)
        total_forces = np.sum(counts)

        # Flat indices for force assignment
        all_indices = self._rng.choice(
            total_elements,
            size=total_forces,
            replace=True,  # reuse allowed across different samples
        )

        # Random force values
        all_forces = self._rng.uniform(force_min, force_max, size=total_forces)
        for i in range(total_forces):
            all_forces[i] = all_forces[i] if self._rng.choice([True, False]) else -all_forces[i]

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

    def triangular(self, num_samples: int, force_max: float = 10.0, force_min: float = 0.1) -> np.ndarray:
        """Generate triangular force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            # Randomly choose a peak position and height
            peak_position = self._rng.integers(0, self._profile_length)
            peak_height = self._rng.uniform(force_min, force_max)
            peak_height = peak_height if self._rng.choice([True, False]) else -peak_height
            side = self._rng.choice([0, 1, 2])

            # Create a triangular profile
            for j in range(self._profile_length):
                if j < peak_position:
                    profiles[i, side, j] = (peak_height / peak_position) * j
                elif j > peak_position:
                    profiles[i, side, j] = (
                        peak_height / (self._profile_length - 1 - peak_position)) * (self._profile_length - 1 - j)
                else:
                    profiles[i, side, j] = peak_height
        return profiles

    def square(self, num_samples: int, force_max: float = 10.0, force_min: float = 0.1) -> np.ndarray:
        """Generate square force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            position_1: int = self._rng.integers(0, self._profile_length).item()
            position_2: int = self._rng.integers(0, self._profile_length).item()
            start_position = min(position_1, position_2)
            end_position = max(position_1, position_2)
            height = self._rng.uniform(force_min, force_max)
            height = height if self._rng.choice([True, False]) else -height
            side = self._rng.choice([0, 1, 2])
            profiles[i, side, start_position:end_position+1] = height
        return profiles

    def gaussian(self, num_samples: int, force_max: float = 10.0, force_min: float = 0.1) -> np.ndarray:
        """Generate Gaussian force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            mean_position = self._rng.integers(0, self._profile_length)
            std_dev = self._rng.uniform(1, self._profile_length / 10)
            height = self._rng.uniform(force_min, force_max)
            height = height if self._rng.choice([True, False]) else -height
            side = self._rng.choice([0, 1, 2])

            x = np.arange(self._profile_length)
            gaussian_profile = height * np.exp(
                -((x - mean_position) ** 2) / (2 * std_dev**2),
            )
            profiles[i, side, :] = gaussian_profile
        return profiles

    def ramp(self, num_samples: int, force_max: float = 10.0, force_min: float = 0.1) -> np.ndarray:
        """Generate ramp force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            position_1 = self._rng.integers(0, self._profile_length).item()
            position_2 = self._rng.integers(0, self._profile_length).item()
            start_position = min(position_1, position_2)
            end_position = max(position_1, position_2)
            height = self._rng.uniform(force_min, force_max)
            height = height if self._rng.choice([True, False]) else -height
            side = self._rng.choice([0, 1, 2])
            direction = self._rng.choice([-1, 1])

            if start_position == end_position:
                profiles[i, side, start_position] = height
            elif direction == -1:
                for j in range(self._profile_length):
                    if j <= start_position or j > end_position:
                        profiles[i, side, j] = 0
                    elif j == end_position:
                        profiles[i, side, j] = height
                    else:
                        profiles[i, side, j] = (
                            height / (end_position - start_position)
                        ) * (j - start_position)
            else:
                for j in range(self._profile_length):
                    if j < start_position or j >= end_position:
                        profiles[i, side, j] = 0
                    elif j == start_position:
                        profiles[i, side, j] = -height
                    else:
                        profiles[i, side, j] = height / (end_position - start_position) * (j - end_position)
        return profiles

    def merger(self, num_samples: int, scaling: float = 10.0) -> np.ndarray:
        """Generate merged force profiles from different shapes."""
        profile_count = self._log_normal_sampling(mean=0.5, sigma=0.8, size=num_samples, rng=self._rng).astype(int)
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            if profile_count[i] == 0:
                profile_count[i] = 1  # Ensure at least one profile is added
            for _ in range(profile_count[i]):
                shape_type = self._rng.choice(
                    ["triangular", "square", "gaussian", "impulse", "ramp"],
                    p=[0.2, 0.2, 0.2, 0.2, 0.2],
                )
                if shape_type == "triangular":
                    profiles[i] += self.triangular(1, scaling)[0]
                elif shape_type == "square":
                    profiles[i] += self.square(1, scaling)[0]
                elif shape_type == "gaussian":
                    profiles[i] += self.gaussian(1, scaling)[0]
                elif shape_type == "impulse":
                    profiles[i] += self.impulse(1, force_max=scaling)[0]
                elif shape_type == "ramp":
                    profiles[i] += self.ramp(1, scaling)[0]

            # Compute energy (L2 norm squared) and scale toward log uniform distribution
            target_energy = self._log_uniform_sampling()
            actual_energy = np.sum(np.square(profiles[i,0,:])) + np.sum(np.square(profiles[i,1,:])) + np.sum(np.square(profiles[i,2,:]))
            if actual_energy == 0:
                logger.warning(f"Sample {i} has zero energy; skipping scaling. {profiles[i]}")
                continue
            scaling_factor = np.sqrt(target_energy / actual_energy)
            profiles[i] *= scaling_factor
        return profiles

    def triangular_only(self, num_samples: int) -> np.ndarray:
        """Generate merged force profiles from different shapes."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            profiles[i] += self.triangular(1, 1)[0]

            # Compute energy (L2 norm squared) and scale toward log uniform distribution
            target_energy = self._log_uniform_sampling()
            actual_energy = np.sum(np.square(profiles[i,0,:])) + np.sum(np.square(profiles[i,1,:])) + np.sum(np.square(profiles[i,2,:]))
            if actual_energy == 0:
                logger.warning(f"Sample {i} has zero energy; skipping scaling. {profiles[i]}")
                continue
            scaling_factor = np.sqrt(target_energy / actual_energy)
            profiles[i] *= scaling_factor
        return profiles

    def _log_uniform_sampling(self, low: float = 1e2, high: float = 5e4, size: int = 1, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
        """Sample from a log-uniform distribution between low and high."""
        log_low = np.log(low)
        log_high = np.log(high)
        return np.exp(rng.uniform(log_low, log_high, size=size))

    def _log_normal_sampling(self, mean: float = 0.0, sigma: float = 1.0, size: int = 1, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
        """Sample from a log-normal distribution with given mean and sigma."""
        return rng.lognormal(mean, sigma, size=size)


if __name__ == "__main__":
    # Example usage
    import logging

    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    generator = ForceProfileGenerator(profile_length=10, batch_seed=42)
    force_profiles = generator.merger(num_samples=100_000, scaling=10.0)
    #force_profiles = generator.triangular_only(num_samples=100_000)
    force_profile_energy = np.sum(force_profiles**2, axis=(1, 2))

    # Find empty profiles
    empty_profiles = np.where(force_profile_energy == 0)[0]
    if len(empty_profiles) > 0:
        logger.warning(f"Found {len(empty_profiles)} empty force profiles at indices: {empty_profiles}")

    logger.info(f"Generated profiles like: {force_profiles[0]} and {force_profiles[1]}")

    # Find the maximum force value
    max_force_value = np.max(np.abs(force_profiles))
    logger.info(f"Maximum force value across all profiles: {max_force_value}")

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.hist(force_profiles.flatten(), bins=200)
    plt.title("Distribution of Force Profiles")
    plt.yscale("log")
    plt.xlabel("Force Profile Value")
    plt.ylabel("Frequency")

    # Logarithmic histogram of energy values
    plt.subplot(2, 2, 2)
    bins = np.logspace(-2, 5, 200).tolist()
    plt.hist(force_profile_energy, bins=bins)
    plt.title("Distribution of Force Profile Energy")
    plt.xscale("log")
    plt.xlabel("Force Profile Energy Value")
    plt.ylabel("Frequency")

    # Test location distribution of forces
    plt.subplot(2, 2, 3)
    mean = np.zeros(force_profiles.shape[1] * force_profiles.shape[2])
    std = np.zeros(force_profiles.shape[1] * force_profiles.shape[2])
    for i in range(force_profiles.shape[1]):
        for k in range(force_profiles.shape[2]):
            mean[i*10 + k] = np.mean(force_profiles[:, i, k])
            std[i*10 + k] = np.std(force_profiles[:, i, k])
    plt.errorbar(
        np.arange(force_profiles.shape[1] * force_profiles.shape[2]),
        mean,
        yerr=std,
        fmt="o",
        ecolor="red",
        capsize=2,
    )
    plt.title("Mean and Std Dev of Force Profile Locations")
    plt.xlabel("Force Profile Index")
    plt.ylabel("Mean Force Value")
    plt.show()
