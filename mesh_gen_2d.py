from dolfin import *
from mshr import Polygon, generate_mesh
import matplotlib.pyplot as plt

meshes=[10,20,50,100,150,200,250,300]
for i in range(len(meshes)):
	domain = Polygon([Point(0,0),Point(1.5,0),Point(2,0.25),Point(3,0.25),Point(3.5,0),Point(5,0),Point(5,1),Point(3.5,1),Point(3,0.75),Point(2,0.75),Point(1.5,1),Point(0,1)])
	mesh = generate_mesh(domain,meshes[i])

	with cpp.io.XDMFFile("mesh2d_"+str(i)+".xdmf") as file:
				file.write(mesh)
			


