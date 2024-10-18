import numpy as np
import scipy
import math


import ufl
from dolfinx import fem, io, mesh, plot, geometry
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner
from basix.ufl import element

from mpi4py import MPI
from petsc4py import PETSc

with io.XDMFFile(MPI.COMM_WORLD, "results/frac_uh_final_116642_0.5.xdmf", "r") as xdmf:
		msh = xdmf.read_mesh()

facet_indices, facet_markers =[], []
facetsx = mesh.locate_entities(msh, msh.topology.dim - 1,lambda x: np.logical_or(np.isclose(x[0], 5),np.isclose(x[0], 0)))
facetsy = mesh.locate_entities(msh, msh.topology.dim - 1,lambda x: np.logical_or(np.isclose(x[1], 0),np.isclose(x[1], 1)))
facetslin1 = mesh.locate_entities(msh, msh.topology.dim - 1, lambda x: np.logical_and(np.logical_and(x[0]>= 1.5,x[0]<=2),np.isclose(x[1],1-0.5*(x[0]-1.5))))
facetslin2 = mesh.locate_entities(msh, msh.topology.dim - 1, lambda x: np.logical_and(np.logical_and(x[0]>= 3,x[0]<=3.5),np.isclose(x[1],0.25-0.5*(x[0]-3))))
facetslin3 = mesh.locate_entities(msh, msh.topology.dim - 1, lambda x: np.logical_and(np.logical_and(x[0]>= 3,x[0]<=3.5),np.isclose(x[1],0.75+0.5*(x[0]-3))))
facetslin4 = mesh.locate_entities(msh, msh.topology.dim - 1, lambda x: np.logical_and(np.logical_and(x[0]>= 1.5,x[0]<=2),np.isclose(x[1],0+0.5*(x[0]-1.5))))
facetsnarrow1=mesh.locate_entities(msh, msh.topology.dim - 1, lambda x: np.logical_and(np.logical_and(x[0]>= 2,x[0]<=3),np.isclose(x[1],0.75)))
facetsnarrow2=mesh.locate_entities(msh, msh.topology.dim - 1, lambda x: np.logical_and(np.logical_and(x[0]>= 2,x[0]<=3),np.isclose(x[1],0.25)))
facet_indices.append(facetsx)
facet_indices.append(facetsy)
facet_indices.append(facetslin1)
facet_indices.append(facetslin2)
facet_indices.append(facetslin3)
facet_indices.append(facetslin4)
facet_indices.append(facetsnarrow1)
facet_indices.append(facetsnarrow2)
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers.append(np.full_like(facetsx, 1))
facet_markers.append(np.full_like(facetsy, 1))
facet_markers.append(np.full_like(facetslin1, 1))
facet_markers.append(np.full_like(facetslin2, 1))
facet_markers.append(np.full_like(facetslin3, 1))
facet_markers.append(np.full_like(facetslin4, 1))
facet_markers.append(np.full_like(facetsnarrow1, 1))
facet_markers.append(np.full_like(facetsnarrow2, 1))

facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(msh, msh.topology.dim - 1, facet_indices[sorted_facets], facet_markers[sorted_facets])    	


smsh,tr,tr2,tr3 = mesh.create_submesh(msh,1,facet_indices)

msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
with io.XDMFFile(msh.comm, "facets_test.xdmf", "w") as file:
		file.write_mesh(smsh)
		#file.write_meshtags(facet_tag,msh.geometry)