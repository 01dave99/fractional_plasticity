import numpy as np
import scipy

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from basix.ufl import element

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

dim=2
y0=10000
mu=5500
kappa=12070
eps=0.1


##Specification of return mapping:

def f(sigma):
	return(ufl.sqrt(inner(ufl.dev(sigma),ufl.dev(sigma)))-y0)

def eval_R_1(sigma):
		dl=f(sigma)/(2*mu)
		sigma_new=sigma-dl*(2*mu*ufl.dev(sigma))/ufl.sqrt(inner(ufl.dev(sigma),ufl.dev(sigma)))
		
		return(sigma_new)

def eval_R_2(sigma):
		dl=f(sigma)/(2*mu)
		sigma_new=sigma-dl*(2*mu*ufl.dev(sigma))/ufl.sqrt(inner(ufl.dev(sigma),ufl.dev(sigma)))
		
		return(dl)


		
def eval_S(sigma_tr,sigma,lamb):
		id4=ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l))
		tmp1=(id4-1/dim*(ufl.outer(ufl.Identity(dim),ufl.Identity(dim))))/ufl.sqrt(inner(ufl.dev(sigma_tr),ufl.dev(sigma_tr)))
		tmp2=ufl.outer(ufl.dev(sigma_tr),ufl.dev(sigma_tr))/(ufl.sqrt(inner(ufl.dev(sigma_tr),ufl.dev(sigma_tr)))**3)
		dl=ufl.dev(sigma_tr)/(ufl.sqrt(inner(ufl.dev(sigma_tr),ufl.dev(sigma_tr)))*2*mu)
		S=id4-lamb*2*mu*(tmp1-tmp2)-2*mu*ufl.outer(dl*2*mu,dl)
		return(S)




#Finite Element specifications:

#mesh
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(64, 64),
                            cell_type=mesh.CellType.triangle)

#function space:
el=ufl.VectorElement("Lagrange",ufl.triangle,1)
V=fem.functionspace(msh, el)

facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 1.0)))
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bc = fem.dirichletbc(value=np.zeros(2), dofs=dofs, V=V)

#functions:
x = ufl.SpatialCoordinate(msh)

b1=0
b2=-10000
b=ufl.as_vector([b1,b2])
t1=0
t2=10000*ufl.exp(-(x[0]-0.5)**2)
t=ufl.as_vector([t1,t2])
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)
L=inner(b,v) * dx + inner(t,v)* ds

for m in range(100):
	if m==0:
		eu=0.5*(grad(u)+ufl.transpose(grad(u)))
		ev=0.5*(grad(v)+ufl.transpose(grad(v)))
		a=inner(ufl.dev(eu)*2*mu+kappa*ufl.tr(eu)*ufl.Identity(dim),ev)*dx
		problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		uh = problem.solve()
		
	else:
		etr=0.5*(grad(uh)+ufl.transpose(grad(uh)))
		sigma_tr=ufl.dev(etr)*2*mu+kappa*ufl.tr(etr)*ufl.Identity(dim)

		#stress return mapping
		sigma=ufl.conditional(ufl.le(f(sigma_tr),0),sigma_tr,eval_R_1(sigma_tr))
		dl=ufl.conditional(ufl.le(f(sigma_tr),0),0,eval_R_2(sigma_tr))


		#residual
		ev=0.5*(grad(v)+ufl.transpose(grad(v)))
		r=inner(sigma,ev)*dx-L
		rvec=fem.petsc.assemble_vector(fem.form(r))
		fem.petsc.set_bc(rvec,bcs=[bc])
		print("Residual norm: "+str(rvec.norm()))
		if rvec.norm()<eps: break

		i, j, k, l = ufl.indices(4)
		
		#subgradient
		S=ufl.conditional(ufl.eq(dl,0),ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l)),eval_S(sigma_tr,sigma,dl))

		eu=0.5*(grad(u)+ufl.transpose(grad(u)))
		tmp=ufl.dev(eu)*2*mu+kappa*ufl.tr(eu)*ufl.Identity(dim)
		tmp2=ufl.as_tensor(S[i,j,k,l]*tmp[k,l],(i,j))
		a= inner(tmp2,ev) * dx
		
		problem = LinearProblem(a, -r, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		duh = problem.solve()
		
		uh.x.array[:]=uh.x.array[:]+duh.x.array[:]

		with io.XDMFFile(msh.comm, "test"+str(m)+".xdmf", "w") as file:
			file.write_mesh(msh)
			file.write_function(uh)
		
