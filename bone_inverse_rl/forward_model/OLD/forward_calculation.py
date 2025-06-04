from fenics import Function, grad, div, Identity, inner, project, near

class ForwardModel:
    """
    ForwardModel Class

    This class represents a forward model for simulating material behavior based on density and stress-strain relationships. 
    It provides methods to calculate material properties, stress, strain, strain energy density, and density changes.

    Attributes:
        V_ele (dolfin.FunctionSpace): The finite element function space for the model.
        V (dolfin.FunctionSpace): The function space for the density field.
        rho_min (float): Minimum density value.
        rho_max (float): Maximum density value.
        cnt_cell_converged (list): List tracking convergence status for each cell.
        M (float): Material constant used in the calculation of Young's modulus.
        gamma (float): Exponent used in the calculation of Young's modulus.
        nu (float): Poisson's ratio.
        B (float): Constant used in density change calculation.
        k (float): Reference stimulus value.
        dt (float): Time step for density update.
        F (dolfin.Function): External force applied to the model.
        ds (dolfin.Measure): Measure for boundary integration.

    Methods:
        calculate_E(updated_rho_val):
            Calculates the updated Young's modulus (E) based on the given density values.

        calculate_Lame_coefficients(E_val):
            Computes the Lame coefficients (mu and lambda) based on the Young's modulus.

        epsilon(u):
            Computes the strain tensor for a given displacement field.

        sigma(u, mu, lmbda):
            Computes the stress tensor for a given displacement field, shear modulus, and first Lame parameter.

        calculate_SED(epsilon_val, sigma_val):
            Calculates the strain energy density (SED) based on the strain and stress tensors.

        calculate_Density_change(rho_vals, SED):
            Updates the density values based on the strain energy density and stimulus, while ensuring convergence criteria.
    """
    def __init__(self, V_ele, V, rho_min, rho_max, cnt_cell_converged, M, gamma, nu, B, k, dt, F, ds):
        self.V_ele = V_ele
        self.V = V
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.cnt_cell_converged = cnt_cell_converged
        self.M = M
        self.gamma = gamma
        self.nu = nu
        self.B = B
        self.k = k
        self.dt = dt
        self.F = F
        self.ds = ds

    def calculate_E(self, updated_rho_val):
        E_updated = Function(self.V_ele)
        E_array = E_updated.vector().get_local()
        for i, rho in enumerate(updated_rho_val):
            E_array[i] = self.M * pow(rho, self.gamma)
        E_updated.vector().set_local(E_array)
        return E_updated

    def calculate_Lame_coefficients(self, E_val):
        mu_val = E_val / (2 * (1 + self.nu))
        lmbda_val = (E_val * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu_val, lmbda_val

    def epsilon(self, u):
        strain_u = 0.5 * (grad(u) + grad(u).T)
        return strain_u

    def sigma(self, u, mu, lmbda):
        stress_u = lmbda * div(u) * Identity(u.geometric_dimension()) + 2 * mu * self.epsilon(u)
        return stress_u

    def calculate_SED(self, epsilon_val, sigma_val):
        SED_val = 0.5 * inner(sigma_val, epsilon_val)
        SED_plot = project(SED_val, self.V_ele)
        SED_values = SED_plot.vector().get_local()
        return SED_values, SED_plot

    def calculate_Density_change(self, rho_vals, SED):
        change_in_density = []
        tol = 1E-6
        rho_plot = Function(self.V_ele)
        rho_array = rho_plot.vector().get_local()
        stimulus = rho_plot.vector().get_local()
        for i, SED_val in enumerate(SED):
            if self.cnt_cell_converged[i] == 0:
                stimulus[i] = SED_val / rho_vals[i]
                change_in_density.append(self.B * (stimulus[i] - self.k))
                rho_array[i] = rho_vals[i] + self.dt * change_in_density[i]

                if rho_array[i] <= self.rho_min:
                    self.cnt_cell_converged[i] = 1
                    rho_array[i] = self.rho_min

                elif rho_array[i] >= self.rho_max:
                    self.cnt_cell_converged[i] = 1
                    rho_array[i] = self.rho_max

                elif near(change_in_density[i], 0.0, tol):
                    self.cnt_cell_converged[i] = 1
            else:
                change_in_density.append(0)
                rho_array[i] = rho_vals[i]
                stimulus[i] = SED_val / rho_vals[i]

            rho_plot.vector().set_local(rho_array)

        return rho_plot, rho_array, self.cnt_cell_converged
