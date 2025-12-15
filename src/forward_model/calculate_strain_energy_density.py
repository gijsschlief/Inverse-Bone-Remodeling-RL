"""Calculate the strain energy density (SED) from the displacement field."""

import ufl.tensors  # type: ignore
from fenics import (  # type: ignore
    Function,
    FunctionSpace,
    Identity,
    KrylovSolver,
    PETScMatrix,
    PETScVector,
    TestFunction,
    TrialFunction,
    assemble,
    div,
    dx,
    grad,
    inner,
)


class StrainEnergyDensityCalculator:
    """Class to calculate the strain energy density (SED) from the displacement field."""

    def __init__(
        self,
        cell_density_space: FunctionSpace,
        spatial_dimension: int,
        solver_parameters: dict | None = None,
    ) -> None:
        """Initialize the StrainEnergyDensityCalculator.

        Args:
        ----
            cell_density_space (FunctionSpace): The function space for the cell density.
            spatial_dimension (int): The spatial dimension of the problem.
            solver_parameters (dict, optional): Custom solver parameters for the linear variational solver.

        """
        self.spatial_dimension = spatial_dimension

        self.sed_function = Function(cell_density_space)

        sed_trial_function = TrialFunction(cell_density_space)
        self._sed_test_function = TestFunction(cell_density_space)
        strain_energy_density_matrix = PETScMatrix()

        assemble(
            inner(sed_trial_function, self._sed_test_function) * dx,
            tensor=strain_energy_density_matrix,
        )
        self._strain_energy_density_matrix = strain_energy_density_matrix

        self._solver = KrylovSolver(
            solver_parameters["linear_solver"] if solver_parameters else "cg",
            solver_parameters["preconditioner"] if solver_parameters else "hypre_amg",
        )
        if solver_parameters and "krylov_solver" in solver_parameters:
            self._solver.parameters.update(solver_parameters["krylov_solver"])
        self._solver.set_operators(
            strain_energy_density_matrix,
            strain_energy_density_matrix,
        )

    @staticmethod
    def _calculate_strain_tensor(displacement: Function) -> ufl.tensors.ListTensor:
        """Calculate the strain tensor from the displacement field."""
        return 0.5 * (grad(displacement) + grad(displacement).T)

    def _calculate_stress_tensor(
        self,
        displacement: Function,
        strain_tensor: ufl.tensors.ListTensor,
        lame_function: Function,
        shear_function: Function,
    ) -> ufl.tensors.ListTensor:
        """Calculate the stress tensor using the strain tensor, shear_modules and first Lame coefficient (lambda)."""
        return (
            lame_function * div(displacement) * Identity(self.spatial_dimension)
            + 2 * shear_function * strain_tensor
        )

    def calculate(
        self,
        displacement: Function,
        lame_function: Function,
        shear_function: Function,
    ) -> Function:
        """Calculate the strain energy density (SED) from the strain and stress tensors.

        Args:
        ----
            displacement (Function): The displacement field.
            lame_function (Function): The first Lame coefficient (lambda).
            shear_function (Function): The shear modulus (mu).

        Returns:
        -------
            Function: The calculated strain energy density function.

        """
        strain_tensor = self._calculate_strain_tensor(displacement)
        stress_tensor = self._calculate_stress_tensor(
            displacement,
            strain_tensor,
            lame_function,
            shear_function,
        )

        sed_expression = 0.5 * inner(stress_tensor, strain_tensor)
        linear_sed_form = PETScVector()
        assemble(
            inner(sed_expression, self._sed_test_function) * dx,
            tensor=linear_sed_form,
        )

        self._solver.solve(self.sed_function.vector(), linear_sed_form)
        return self.sed_function


def example_usage() -> None:
    """StrainEnergyDensityCalculator example usage."""
    import logging  # noqa: PLC0415
    import timeit  # noqa: PLC0415

    from fenics import Constant, UnitSquareMesh, VectorFunctionSpace  # noqa: PLC0415

    logger = logging.getLogger(__name__)

    mesh = UnitSquareMesh(10, 10, "left")

    displacement_vector_space = VectorFunctionSpace(mesh, "P", 2)
    density_function_space = FunctionSpace(mesh, "DG", 0)

    displacement = Function(displacement_vector_space)
    displacement.assign(Constant((0.1, 0.2)))

    lame_function = Function(density_function_space)
    lame_function.assign(Constant(2.0))

    shear_function = Function(density_function_space)
    shear_function.assign(Constant(1.0))

    sed_calculator = StrainEnergyDensityCalculator(
        cell_density_space=density_function_space,
        spatial_dimension=2,
    )
    start_time = timeit.default_timer()
    sed = sed_calculator.calculate(displacement, lame_function, shear_function)
    elapsed_time = timeit.default_timer() - start_time

    start_time2 = timeit.default_timer()
    sed = sed_calculator.calculate(displacement, lame_function, shear_function)
    elapsed_time2 = timeit.default_timer() - start_time2

    logging.basicConfig(level=logging.INFO)
    logger.info(
        f"First calculation took {elapsed_time:.6f} seconds, second took {elapsed_time2:.6f} seconds",
    )
    logger.info("Min/Max SED: %.6f / %.6f", sed.vector().min(), sed.vector().max())


if __name__ == "__main__":
    example_usage()
