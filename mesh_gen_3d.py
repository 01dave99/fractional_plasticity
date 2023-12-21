from dolfin import *
from mshr import Sphere, Box, Cylinder, generate_mesh
from math import pi, sin, cos, sqrt

# Create geometry
sbox = Box(Point(0, 0, 0), Point(8, 2, 2))
cylinder1 = Cylinder(Point(3, -1, -1),Point(3, 3,3), 0.5, 0.5)
cylinder2 = Cylinder(Point(5, 1, 3),Point(5, 1,-3), 0.5, 0.5)
cylinder3 = Cylinder(Point(6.5, -3, 1),Point(6.5, 3,1), 0.5, 0.5)
cylinder4 = Cylinder(Point(1.5, -1, 3),Point(1.5, 3,-1), 0.5, 0.5)
geometry=sbox-cylinder1-cylinder2-cylinder3-cylinder4


# Create mesh
mesh = generate_mesh(geometry, 50)
with cpp.io.XDMFFile("mesh3d.xdmf") as file:
				file.write(mesh)