from fenics import *
import numpy as np
import time

class DensitySimulation:
    def __init__(self, force_profile, initial_density, time_steps, dt, parameters):
        """
        Initialize the density simulation with parameters and setup.
        :param force_profile: Force profile matrix.
        :param initial_density: Initial bone density matrix.
        :param time_steps: Number of time steps for the simulation.
        :param dt: Time step size.
        :param parameters: Dictionary containing simulation parameters.
        """
        self.initialize_parameters(force_profile, initial_density, time_steps, dt, parameters)
        self.setup_mesh_and_spaces()
        self.initialize_density()
        self.setup_boundary_conditions()
        self.setup_subdomains()
        self.F = Expression("m*x[0]+c", m=-10, c=10, degree=1)
        self.E0 = self.calculate_E(self.rho_val)
        self.mu, self.lmbda = self.calculate_Lame_coefficients(self.E0)
        self.u = Function(self.V)

    def initialize_parameters(self, force_profile, initial_density, time_steps, dt, parameters):
        self.time_steps = time_steps  # Store time_steps as an instance variable
        # Initialize density and force profiles
        self.density_profile = initial_density
        self.force_profile = force_profile
        self.dt = dt
        self.T = time_steps * dt
        self.parameters = parameters

        # Data extraction from the density profile
        self.X = self.density_profile.shape[0]  # Number of rows in initial density
        self.Y = self.density_profile.shape[1]  # Number of columns in initial density
        self.rho0 = self.density_profile.mean()  # Initial average density
        
        # Data extraction from parameters
        self.file_location = parameters.get('file_location', 'data/')  # Location of the data files
        self.rho_min = parameters.get('rho_min', 0.1)  # Minimum bone density
        self.rho_max = parameters.get('rho_max', 1.5)  # Maximum bone density
        self.tolerance = parameters.get('tolerance', 1E-14)  # Tolerance for convergence
        self.B = parameters.get('B', 0.1)  # Coefficient for density change
        self.k = parameters.get('k', 0.01)  # Threshold for density change
        self.nu = parameters.get('nu', 0.3)  # Poisson's ratio
        self.M = parameters.get('M', 1.0)  # Modulus of elasticity
        self.gamma = parameters.get('gamma', 2.0)  # Exponent for density elasticity
        self.file_name = parameters.get('file_name', 'density_simulation')  # Base name for output files
        self.file_extension = parameters.get('file_extension', '.pvd')  # File extension for output files

    def setup_mesh_and_spaces(self):
        self.mesh = UnitSquareMesh(self.X, self.Y, 'left')
        self.V = VectorFunctionSpace(self.mesh, "P", 1)
        self.V_ele = FunctionSpace(self.mesh, "DG", 0)
        self.d = self.V.ufl_element().value_shape()[0]
        self.f = Constant((0, 0))
        self.v = TestFunction(self.V)
        self.u_trial = TrialFunction(self.V)
        self.cnt_cells = self.mesh.num_cells()

    def initialize_density(self):
        self.rho_val = [self.rho0 for _ in range(self.mesh.num_cells())]
        self.updated_rho_val = self.rho_val[:]
        self.cnt_cell_converged = [0 for _ in range(self.mesh.num_cells())]

    def setup_boundary_conditions(self):
        def bottom_fixed_boundary(x, on_boundary):
            return near(x[0], 0, self.tolerance) and near(x[1], 0, self.tolerance)

        def bottom_right_boundary(x, on_boundary):
            return near(x[1], 0, self.tolerance) and x[0] > 0

        bc_fixed = DirichletBC(self.V, Constant((0., 0.)), bottom_fixed_boundary, method='pointwise')
        bc_roller = DirichletBC(self.V.sub(1), Constant(0), bottom_right_boundary)
        self.bcs = [bc_fixed, bc_roller]

    def setup_subdomains(self):
        class Top(SubDomain):
            def __init__(self, tolerance=1E-14):
                super().__init__()
                self.tolerance = tolerance
            def inside(self, x, on_boundary):
                return near(x[1], 1, self.tolerance)

        self.boundaries = MeshFunction('size_t', self.mesh, 1)
        self.boundaries.set_all(0)
        Top().mark(self.boundaries, 1)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
 
    def calculate_E(self, rho_vals):
        E_func = Function(self.V_ele)
        E_array = E_func.vector().get_local()
        for i, rho in enumerate(rho_vals):
            E_array[i] = self.M * pow(rho, self.gamma)
        E_func.vector().set_local(E_array)
        return E_func

    def calculate_Lame_coefficients(self, E_val):
        mu = E_val / (2 * (1 + self.nu))
        lmbda = (E_val * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu, lmbda

    def epsilon(self, u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(self, u, mu, lmbda):
        return lmbda * div(u) * Identity(self.d) + 2 * mu * self.epsilon(u)

    def calculate_SED(self, epsilon_val, sigma_val):
        SED_val = 0.5 * inner(sigma_val, epsilon_val)
        SED_plot = project(SED_val, self.V_ele)
        return SED_plot.vector().get_local(), SED_plot

    def calculate_Density_change(self, rho_vals, SED):
        rho_plot = Function(self.V_ele)
        rho_array = rho_plot.vector().get_local()
        stimulus = np.zeros_like(rho_array)
        change_in_density = []

        for i, SED_val in enumerate(SED):
            if self.cnt_cell_converged[i] == 0:
                stimulus[i] = SED_val / rho_vals[i]
                change = self.B * (stimulus[i] - self.k)
                new_rho = rho_vals[i] + self.dt * change

                if new_rho <= self.rho_min:
                    new_rho = self.rho_min
                    self.cnt_cell_converged[i] = 1
                elif new_rho >= self.rho_max:
                    new_rho = self.rho_max
                    self.cnt_cell_converged[i] = 1
                elif abs(change) < 1E-6:
                    self.cnt_cell_converged[i] = 1

                rho_array[i] = new_rho
                change_in_density.append(change)
            else:
                rho_array[i] = rho_vals[i]
                stimulus[i] = SED_val / rho_vals[i]
                change_in_density.append(0)

        rho_plot.vector().set_local(rho_array)
        return rho_plot, list(rho_array), self.cnt_cell_converged

    def run(self):
        self.start_time = time.time()
        t = 0
        while t <= self.T:
            a = 2 * self.mu * inner(self.epsilon(self.u_trial), self.epsilon(self.v)) * dx \
                + self.lmbda * dot(div(self.u_trial), div(self.v)) * dx
            L = dot(self.f, self.v) * dx + self.v[1] * self.F * self.ds(1)
            solve(a == L, self.u, self.bcs)

            eps_val = self.epsilon(self.u)
            sig_val = self.sigma(self.u, self.mu, self.lmbda)

            SED, _ = self.calculate_SED(eps_val, sig_val)
            rho_func, self.updated_rho_val, self.cnt_cell_converged = self.calculate_Density_change(self.updated_rho_val, SED)

            if t == self.T or sum(self.cnt_cell_converged) == self.cnt_cells:
                File(self.file_name + str(t) + self.file_extension) << rho_func
                if sum(self.cnt_cell_converged) == self.cnt_cells:
                    break

            self.E0.assign(self.calculate_E(self.updated_rho_val))
            self.mu, self.lmbda = self.calculate_Lame_coefficients(self.E0)
            t += self.dt

    def get_final_density(self):
        """
        Get the final density profile after the simulation.
        :return: Final density profile as a NumPy array.
        """

        half_size = len(self.updated_rho_val) // 2
        return np.array(self.updated_rho_val[:half_size]).reshape((self.X, self.Y))
