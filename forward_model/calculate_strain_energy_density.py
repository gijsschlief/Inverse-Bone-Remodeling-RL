#TODO: Create a class that calculates the strain energy density instead of doing it in the density simulation class.
# This will make the code cleaner and more modular.
# The class should take the density function and the mesh as inputs and provide a method to calculate the strain energy density.
# It should also handle the material properties and any necessary updates.
# This will allow for better separation of concerns and easier testing of the strain energy density calculation.

class StrainEnergyDensityCalculator:
    """Class to calculate the strain energy density (SED) from the displacement field."""

    def __init__(self, displacement: Function, mesh: Mesh, lame_function: Function, shear_function: Function):
        """Initialize the calculator with the displacement field and material properties."""
        self.displacement = displacement
        self.mesh = mesh
        self.lame_function = lame_function
        self.shear_function = shear_function
        self.spatial_dimension = mesh.topology().dim()
        self.test_function_density = TestFunction(FunctionSpace(mesh, "CG", 1))
        self.sed_function = Function(FunctionSpace(mesh, "CG", 1))
        self.projection_solver_parameters = {"linear_solver": "cg", "preconditioner": "hypre_amg"}

    def calculate(self) -> None:
        """Calculate the strain energy density."""
        self._calculate_strain_energy_density()

    @staticmethod
    def _calculate_strain_tensor(displacement: Function) -> ufl.tensors.ListTensor:
        """Calculate the strain tensor from the displacement field."""
        strain_tensor = 0.5 * (grad(displacement) + grad(displacement).T)
        return strain_tensor

    def _calculate_stress_tensor(
        self,
        displacement: Function,
        strain_tensor: ufl.tensors.ListTensor,
    ) -> ufl.tensors.ListTensor:
        """Calculate the stress tensor using the strain tensor, shear_modules and first Lame coefficient (lambda)."""
        stress_tensor = (
            self.lame_function * div(displacement) * Identity(self.spatial_dimension)
            + 2 * self.shear_function * strain_tensor
        )
        return stress_tensor

    def _calculate_strain_energy_density(self) -> None:
        """Calculate the strain energy density (SED) from the strain and stress tensors."""
        strain_tensor = self._calculate_strain_tensor(self.displacement)
        stress_tensor = self._calculate_stress_tensor(self.displacement, strain_tensor)

        sed_expression = 0.5 * inner(stress_tensor, strain_tensor)
        linear_sed_form = inner(sed_expression, self.test_function_density) * dx
        solve(
            self.strain_energy_density_form == linear_sed_form,
            self.sed_function,
            solver_parameters=self.projection_solver_parameters,
        )
