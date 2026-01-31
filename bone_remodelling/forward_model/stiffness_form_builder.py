"""Form builder for stiffness matrix in bone remodeling simulation."""

import ufl  # type: ignore
from fenics import (  # type: ignore
    Function,
    FunctionSpace,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
)


class StiffnessFormBuilder:
    """Builder for the stiffness form in bone remodeling simulations.

    This class constructs the stiffness form used in the finite element method
    for bone remodeling simulations, based on the shear and Lame functions.
    """

    def __init__(
        self,
        shear_function: Function,
        lame_function: Function,
        displacement_space: FunctionSpace,
        displacement_test_function: Function,
    ) -> None:
        """Initialize the StiffnessFormBuilder with shear and Lame functions."""
        self.shear_function = shear_function
        self.lame_function = lame_function
        self.displacement_trial = TrialFunction(displacement_space)
        self.displacement_test_function = displacement_test_function
        self._initialize_stiffness_form()

    def _initialize_stiffness_form(self) -> None:
        self.stiffness_form = (
            2
            * self.shear_function
            * inner(
                self._calculate_strain_tensor(self.displacement_trial),
                self._calculate_strain_tensor(self.displacement_test_function),
            )
            * dx
            + self.lame_function
            * dot(div(self.displacement_trial), div(self.displacement_test_function))
            * dx
        )

    @staticmethod
    def _calculate_strain_tensor(displacement: Function) -> ufl.tensors.ListTensor:
        """Calculate the strain tensor from the displacement field."""
        return 0.5 * (grad(displacement) + grad(displacement).T)

    def get_stiffness_form(self) -> ufl.form.Form:
        """Return the stiffness form."""
        return self.stiffness_form
