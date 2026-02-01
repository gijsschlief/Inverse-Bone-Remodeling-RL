"""Generates random force profiles for bone remodeling simulations."""

import logging
from collections.abc import Callable

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForceProfileGenerator:
    """Generates random force profiles for bone remodeling simulations."""

    def __init__(
        self,
        profile_top: int = 10,
        profile_sides: int = 10,
        force_bounds: tuple[float, float] = (0.1, 10.0),
        batch_seed: int | None = None,
    ) -> None:
        """Initialize the force profile generator.

        Args:
        ----
            profile_top (int): Resolution of the force profile in the top direction.
            profile_sides (int): Resolution of the force profile in the sides direction.
            force_bounds (tuple[float, float]): Minimum and maximum force values.
            batch_seed (int | None): Seed for random number generation. If None, uses a random seed.

        """
        self._profile_top = profile_top
        self._profile_sides = profile_sides
        self._max_length = max(profile_top, profile_sides)
        self._rng = np.random.default_rng(batch_seed)
        self._force_min = force_bounds[0]
        self._force_max = force_bounds[1]

    def impulse(
        self,
    ) -> np.ndarray:
        """Generate random force profiles on the profile.

        Args:
        ----
            force_count_max (int): Maximum number of forces to apply in each profile.
            force_max (float): Maximum absolute value of the forces.
            force_min (float): Minimum absolute value of the forces.

        """
        total_elements = self._profile_top + 2 * self._profile_sides
        force_profile = np.zeros((3, self._max_length), dtype=float) # 3 sides

        force_count = self._log_normal_sampling(mean=0.5, sigma=0.8)[0].astype(int)
        force_count = min(force_count, total_elements)
        random_location = self._rng.choice(total_elements, size=force_count, replace=False)
        force_magnitude = self._rng.uniform(self._force_min, self._force_max, size=force_count)
        force_sign = self._rng.choice([-1, 1], size=force_count)
        force_value = force_magnitude * force_sign

        for i in range(force_count):
            index = random_location[i]
            if index < self._profile_sides:
                force_profile[0, index] = force_value[i]
            elif index < self._profile_top + self._profile_sides:
                top_index = index - self._profile_sides
                force_profile[1, top_index] = force_value[i]
            else:
                right_index = index - self._profile_top - self._profile_sides
                force_profile[2, right_index] = force_value[i]
        return force_profile

    def triangular_old(self) -> np.ndarray:
        """Generate triangular force profiles."""
        force_profile = np.zeros((3, self._max_length), dtype=float)

        # Randomly choose a peak position and height
        side = self._rng.choice([0, 1, 2])
        profile_length = self._profile_top if side == 1 else self._profile_sides
        peak_position = self._rng.integers(0, profile_length)
        peak_height = self._rng.uniform(self._force_min, self._force_max)
        peak_height = peak_height if self._rng.choice([True, False]) else -peak_height

        # Create a triangular profile
        for j in range(profile_length):
            if j < peak_position:
                force_profile[side, j] = (peak_height / peak_position) * j
            elif j > peak_position:
                force_profile[side, j] = (
                    peak_height / (profile_length - 1 - peak_position)) * (profile_length - 1 - j)
            else:
                force_profile[side, j] = peak_height
        return force_profile

    def triangular(self) -> np.ndarray:
        """Generate triangular force profiles without loops or div-by-zero risk."""
        force_profile = np.zeros((3, self._max_length), dtype=float)
        side = self._rng.choice([0, 1, 2])
        profile_length = self._profile_top if side == 1 else self._profile_sides

        peak_position = self._rng.integers(0, profile_length)
        peak_height = self._rng.uniform(self._force_min, self._force_max)
        peak_height = peak_height if self._rng.choice([True, False]) else -peak_height
        x = np.arange(profile_length)
        xp = [0, peak_position, profile_length - 1]
        fp = [0, peak_height, 0]

        force_profile[side, :profile_length] = np.interp(x, xp, fp)
        return force_profile

    def square(self) -> np.ndarray:
        """Generate square force profiles."""
        force_profile = np.zeros((3, self._max_length), dtype=float)
        side = self._rng.choice([0, 1, 2])
        profile_length = self._profile_top if side == 1 else self._profile_sides
        position_1: int = self._rng.integers(0, profile_length).item()
        position_2: int = self._rng.integers(0, profile_length).item()
        start_position = min(position_1, position_2)
        end_position = max(position_1, position_2)
        height = self._rng.uniform(self._force_min, self._force_max)
        height = height if self._rng.choice([True, False]) else -height
        force_profile[side, start_position:end_position+1] = height
        return force_profile

    def gaussian(self) -> np.ndarray:
        """Generate Gaussian force profiles."""
        force_profile = np.zeros((3, self._max_length), dtype=float)

        side = self._rng.choice([0, 1, 2])
        profile_length = self._profile_top if side == 1 else self._profile_sides
        mean_position = self._rng.integers(0, profile_length)
        std_dev = self._rng.uniform(1, profile_length / 10)
        height = self._rng.uniform(self._force_min, self._force_max)
        height = height if self._rng.choice([True, False]) else -height

        x = np.arange(profile_length)
        gaussian_profile = height * np.exp(
            -((x - mean_position) ** 2) / (2 * std_dev**2),
        )
        force_profile[side, :profile_length] = gaussian_profile
        return force_profile

    def ramp(self) -> np.ndarray:
        """Generate ramp force profiles."""
        force_profile = np.zeros((3, self._max_length), dtype=float)

        side = self._rng.choice([0, 1, 2])
        profile_length = self._profile_top if side == 1 else self._profile_sides

        position_1 = self._rng.integers(0, profile_length).item()
        position_2 = self._rng.integers(0, profile_length).item()
        start_position = min(position_1, position_2)
        end_position = max(position_1, position_2)
        height = self._rng.uniform(self._force_min, self._force_max)
        height = height if self._rng.choice([True, False]) else -height

        if start_position == end_position:
            if end_position < profile_length - 1:
                end_position += 1
            else:
                start_position -= 1

        x = np.arange(profile_length)[start_position:end_position+1]
        xp = [start_position, end_position]
        fp = [0, height] if self._rng.choice([True, False]) else [height, 0]
        ramp = np.interp(x, xp, fp)
        force_profile[side, start_position:end_position+1] = ramp
        return force_profile

    def merger(self, num_samples: int, lower_energy_bound: float = 1e-12) -> np.ndarray:
        """Generate merged force profiles from different shapes."""
        profile_count = self._log_normal_sampling(mean=0.5, sigma=0.8, size=num_samples).astype(int)

        profiles = np.zeros((num_samples, 3, self._max_length), dtype=float)

        shape_generators: dict[str, Callable[[], np.ndarray]] = {
            "triangular": self.triangular,
            "square": self.square,
            "gaussian": self.gaussian,
            "impulse": self.impulse,
            "ramp": self.ramp,
        }
        shape_names = list(shape_generators.keys())

        for i in range(num_samples):
            for _ in range(profile_count[i]):
                shape = self._rng.choice(shape_names)
                profiles[i] += shape_generators[shape]()

            target_energy = self._log_uniform_sampling().item()
            actual_energy = np.sum(profiles[i]**2)
            if actual_energy < lower_energy_bound:
                logger.warning(f"Sample {i} has zero energy; skipping scaling. {profiles[i]}")
                continue
            scaling_factor = np.sqrt(target_energy / actual_energy)
            profiles[i] *= scaling_factor
        return profiles

    def triangular_only(self, num_samples: int, lower_energy_bound: float = 1e-12) -> np.ndarray:
        """Generate merged force profiles from different shapes."""
        profiles = np.zeros((num_samples, 3, self._max_length), dtype=float)
        for i in range(num_samples):
            profiles[i] += self.triangular()

            target_energy = self._log_uniform_sampling().item()
            actual_energy = np.sum(profiles[i]**2)
            if actual_energy < lower_energy_bound:
                logger.warning(f"Sample {i} has zero energy; skipping scaling. {profiles[i]}")
                continue
            scaling_factor = np.sqrt(target_energy / actual_energy)
            profiles[i] *= scaling_factor
        return profiles

    def _log_uniform_sampling(self, low: float = 1e2, high: float = 5e4, size: int = 1) -> np.ndarray:
        """Sample from a log-uniform distribution between low and high."""
        log_low = np.log(low)
        log_high = np.log(high)
        return np.exp(self._rng.uniform(log_low, log_high, size=size))

    def _log_normal_sampling(self, mean: float = 0.0, sigma: float = 1.0, size: int = 1) -> np.ndarray:
        """Sample from a log-normal distribution with given mean and sigma."""
        return np.clip(self._rng.lognormal(mean, sigma, size=size).astype(int), 1, None)

def example_usage() -> None:
    """Use of the ForceProfileGenerator."""
    generator = ForceProfileGenerator(profile_top=10, profile_sides=10, force_bounds=(0.1, 10.0), batch_seed=42)
    force_profiles = generator.merger(num_samples=100_000)
    #force_profiles = generator.triangular_only(num_samples=100_000)
    force_profile_energy = np.sum(force_profiles**2, axis=(1, 2))

    # Find empty profiles
    empty_profiles = np.where(force_profile_energy == 0)[0]
    if len(empty_profiles) > 0:
        logger.warning(f"Found {len(empty_profiles)} empty force profiles at indices: {empty_profiles}")

    logger.info(f"Generated profiles like: {force_profiles[0]} and {force_profiles[1]}")
    max_force_value = np.max(np.abs(force_profiles))
    logger.info(f"Maximum force value across all profiles: {max_force_value}")

    visualise_profiles(force_profiles, force_profile_energy)

def visualise_profiles(force_profiles: np.ndarray, force_profile_energy: np.ndarray) -> None:
    """Visualise generated force profiles."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

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
            mean[i*force_profiles.shape[2] + k] = np.mean(force_profiles[:, i, k])
            std[i*force_profiles.shape[2] + k] = np.std(force_profiles[:, i, k])
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

if __name__ == "__main__":
    example_usage()
