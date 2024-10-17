import meshio
import gmsh
import pygmsh

mesh_size=0.1
geom=pygmsh.occ.Geometry()

model3D=geom.__enter__()
model3D.characteristic_length_max= mesh_size
box= model3D.add_box([0,0,0],[3,1,1])
cyl=model3D.add_cylinder([1.5,-1,0.5],[0,3,0],0.25)
cut=model3D.boolean_difference(box,cyl)
model3D.synchronize()
msh=geom.generate_mesh(dim=3)
meshio.write("mesh3D_new_1.xdmf",meshio.Mesh(points=msh.points, cells={"tetra": msh.get_cells_type("tetra")}))
model3D.__exit__()

