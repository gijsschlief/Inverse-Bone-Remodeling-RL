from fenics import *

class InitialConditions:
    def __init__(self, density_profile, force_profile, dt, parameters):
        """
        Initialize the initial conditions for the simulation.

        :param density_profile: A callable defining the initial density profile.
        :param force_profile: A callable defining the initial force profile.
        :param dt: Time step for the simulation.
        :param parameters: Dictionary containing simulation parameters.
        """
        # Initialize density and force profiles
        self.density_profile = density_profile
        self.force_profile = force_profile
        self.dt = dt
        self.parameters = parameters

        # Data extraction from the density profile
        self.X = density_profile.shape[0]  # Number of rows in initial density
        self.Y = density_profile.shape[1]  # Number of columns in initial density
        self.rho0 = density_profile.mean()  # Initial average density
        
        
        self.tolerance = parameters.get('tolerance', 1E-14)  # Tolerance for convergence

        mesh = UnitSquareMesh(self.X, self.Y)
        self.mesh = mesh

        # Define function spaces
        self.density_space = FunctionSpace(mesh, "CG", 1)
        self.force_space = VectorFunctionSpace(mesh, "CG", 1)

        # Initialize density and force functions
        self.density = Function(self.density_space)
        self.force = Function(self.force_space)


        # Initialize boundaries and other variables
        self.initialize_boundaries()

    def set_initial_density(self):
        """
        Set the initial density profile.
        """
        density_expr = Expression("density_profile", density_profile=self.density_profile, degree=1)
        self.density.interpolate(density_expr)

    def set_initial_force(self):
        """
        Set the initial force profile.
        """
        force_expr = Expression(("force_x", "force_y"), force_x=self.force_profile[0], force_y=self.force_profile[1], degree=1)
        self.force.interpolate(force_expr)

    def apply_initial_conditions(self):
        """
        Apply the initial conditions to the simulation.
        """
        self.set_initial_density()
        self.set_initial_force()

    def initialize_boundaries(self):
        """
        Initialize the boundaries and simulation variables.
        """
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        V_ele = FunctionSpace(self.mesh, "DG", 0)

        # Initialize variables
        self.cnt_cell_converged = []
        self.FName_str = str(self.parameters['file_location'])
        self.rho_val = []
        for cell_s in cells(self.mesh):
            self.rho_val.append(self.rho0)
            self.cnt_cell_converged.append(0)

        # Initialize simulation time
        self.t = self.dt
        self.cnt_cells = self.mesh.num_cells()

        Fixed_left = Constant((0., 0.))
        self.bc_Fixed = DirichletBC(V, Fixed_left, self.bottom_fixed_boundary, method='pointwise')
        Roller_right = Constant(0)
        self.bc_roller = DirichletBC(V.sub(1), Roller_right, self.bottom_right_boundary)

        self.bcs = [self.bc_Fixed, self.bc_roller]

        self.boundaries = MeshFunction('size_t', self.mesh, 1)
        self.boundaries.set_all(0)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
    
    def bottom_fixed_boundary(self, x, _):
        return near(x[0], 0, self.tolerance) and near(x[1], 0, self.tolerance)

    def bottom_right_boundary(self, x, _):
        return near(x[1], 0, self.tolerance) and x[0] > 0
    
    def get_rho_val(self):
        """
        Get the current density values.
        """
        return self.rho_val