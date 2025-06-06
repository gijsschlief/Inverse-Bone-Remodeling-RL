import pyvista as pv

filename = 'Weinans/Density_100.pvd'
reader = pv.get_reader(filename)
reader.set_active_time_point(0)

grid = reader.read()[0]
grid.plot(scalars='f_3514',show_edges=True, show_scalar_bar=True, clim=[0, 1], cpos='xy', show_grid=True)