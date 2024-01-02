import numpy as np
import scipy
import math


import ufl
from dolfinx import fem, io, mesh, plot, geometry
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner
from basix.ufl import element

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

dim=2
y0=10000
mu=55000
kappa=12070
k1=10000
k2=1000
tmax=15000
stepsl=[250,350,500,700,1000]
eps=math.pow(10,-8)


##Specification of return mapping:

def f(sigma,chi1,chi2):
	return(ufl.sqrt(inner(ufl.dev(sigma+chi1),ufl.dev(sigma+chi1)))+chi2-y0)

def epsi(u):
	return(0.5*(grad(u)+ufl.transpose(grad(u))))

def max_0(x):
	return(ufl.conditional(x>=0,x,0))

def eval_gamma(sigma,chi1,chi2):
		dl=max_0(f(sigma,chi1,chi2))/(2*mu+k1+k2)
		return(dl)


		
def eval_S(sigma_tr,chi1,gamma):
		id4=ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l))
		tmp1=(id4-1/dim*(ufl.outer(ufl.Identity(dim),ufl.Identity(dim))))/ufl.sqrt(inner(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1)))
		tmp2=ufl.outer(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1))/(ufl.sqrt(inner(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1))))**3
		ddl=ufl.outer(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1))*2*mu/((2*mu+k1+k2)*inner(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1)))
		S=id4-2*mu*gamma*(tmp1-tmp2)-ddl
		return(S)




#Finite Element specifications:
for steps in stepsl:
	print("Steps: "+str(steps))
	#mesh
	with io.XDMFFile(MPI.COMM_WORLD, "mesh2d_5.xdmf", "r") as xdmf:
	    msh = xdmf.read_mesh()
	#msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
	                            #points=((0.0, 0.0), (5.0, 1.0)), n=(64, 64),
	                            #cell_type=mesh.CellType.triangle)
	bb_tree = geometry.bb_tree(msh, msh.topology.dim)
	#function space:
	el=ufl.VectorElement("Lagrange",ufl.triangle,1)
	elstr=ufl.TensorElement("DG",ufl.triangle,0)
	elsca=ufl.FiniteElement("DG",ufl.triangle,0)

	V=fem.functionspace(msh, el)
	num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
	if msh.comm.rank==0:
		print(num_dofs_global)
	Vstr=fem.functionspace(msh,elstr)
	Vsca=fem.functionspace(msh,elsca)

	facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
	                                       marker=lambda x: np.isclose(x[0], 0.0))
	                                                                     
	dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
	bc = fem.dirichletbc(value=np.zeros(dim), dofs=dofs, V=V)

	facet_indices, facet_markers =[], []
	facets = mesh.locate_entities(msh, msh.topology.dim - 1,lambda x: np.isclose(x[0], 5))
	facet_indices.append(facets)
	facet_indices = np.hstack(facet_indices).astype(np.int32)
	facet_markers.append(np.full_like(facets, 1))
	facet_markers = np.hstack(facet_markers).astype(np.int32)
	sorted_facets = np.argsort(facet_indices)
	facet_tag = mesh.meshtags(msh, msh.topology.dim - 1, facet_indices[sorted_facets], facet_markers[sorted_facets])

	#functions:

	t1vec=np.zeros(steps)
	for i in range(steps):
		t1vec[i]=-abs(2*tmax*(i+1)/steps-tmax)+tmax
	t1vec[steps-1]=0.1
	t2vec=np.zeros(steps)
	time=range(steps)

	residuals=np.zeros((1000,steps))

	u=ufl.TrialFunction(V)
	v=ufl.TestFunction(V)

	uh=fem.Function(V)
	uh_old=fem.Function(V)
	ep=fem.Function(Vstr)
	ep_old=fem.Function(Vstr)
	chi1=fem.Function(Vstr)
	chi1_old=fem.Function(Vstr)
	chi2=fem.Function(Vsca)
	chi2_old=fem.Function(Vsca)
	ds=ufl.Measure("ds",domain=msh,subdomain_data=facet_tag)

	for ti in time:
		#b=ufl.as_vector([b1[ti],b2[ti]])
		t1=t1vec[ti]
		t2=t2vec[ti]
		t=ufl.as_vector([t1,t2])
		L=inner(t,v)* ds(1)


		if ti==0:
			for m in range(1000):
				if m==0:
					a=inner(ufl.dev(epsi(u))*2*mu+kappa*ufl.tr(epsi(u))*ufl.Identity(dim),epsi(v))*dx
					
					problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
					uh=problem.solve()
				else:
					sigma_tr=ufl.dev(epsi(uh))*2*mu+kappa*ufl.tr(epsi(uh))*ufl.Identity(dim)

					#stress return mapping
					dl=eval_gamma(sigma_tr,0*ufl.Identity(dim),0)
					sigma=sigma_tr-2*mu*dl*(ufl.dev(sigma_tr))/ufl.sqrt(inner(ufl.dev(sigma_tr),ufl.dev(sigma_tr)))

					#residual
					r=inner(sigma,epsi(v))*dx-L
					rvec=fem.petsc.assemble_vector(fem.form(r))
					rvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
					fem.petsc.set_bc(rvec,bcs=[bc])
					res=rvec.norm()
					rvec.destroy()
					if msh.comm.rank==0:
						print("Residual norm: "+str(res),flush=True)
						residuals[m,ti]=res
					if res<eps: break

					i, j, k, l = ufl.indices(4)
					
					#subgradient
					S=ufl.conditional(ufl.eq(dl,0),ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l)),eval_S(sigma_tr,0*ufl.Identity(dim),dl))
					
					tmp=ufl.dev(epsi(u))*2*mu+kappa*ufl.tr(epsi(u))*ufl.Identity(dim)
					tmp2=ufl.as_tensor(S[i,j,k,l]*tmp[k,l],(i,j))
					a= inner(tmp2,epsi(v)) * dx
					
					problem = LinearProblem(a, -r, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
					duh = problem.solve()
					
					uh.x.array[:]=uh.x.array[:]+duh.x.array[:]

			ep_f=0.5*(grad(uh)+ufl.transpose(grad(uh)))-ufl.dev(sigma)/(2*mu)-ufl.tr(sigma)*ufl.Identity(dim)/(kappa*dim**2)
			chi1_f=-dl*k1*(ufl.dev(sigma_tr))/ufl.sqrt(inner(ufl.dev(sigma_tr),ufl.dev(sigma_tr)))
			chi2_f=-dl*k2

			epexp=fem.Expression(ep_f, Vstr.element.interpolation_points())
			chi1exp=fem.Expression(chi1_f, Vstr.element.interpolation_points())
			chi2exp=fem.Expression(chi2_f, Vsca.element.interpolation_points())

			ep.interpolate(epexp)
			chi1.interpolate(chi1exp)
			chi2.interpolate(chi2exp)
			
			#with io.XDMFFile(msh.comm, "uh"+str(ti+1)+".xdmf", "w") as file:
			#	file.write_mesh(msh)
			#	file.write_function(uh[ti])
		else:

			ep_old=ep.copy()
			chi1_old=chi1.copy()
			chi2_old=chi2.copy()
			for m in range(1000):
			
				sigma_tr=ufl.dev(epsi(uh)-ep_old)*2*mu+kappa*ufl.tr(epsi(uh)-ep_old)*ufl.Identity(dim)

				#stress return mapping
				dl=eval_gamma(sigma_tr,chi1_old,chi2_old)
				sigma=sigma_tr-2*mu*dl*(ufl.dev(sigma_tr+chi1_old))/ufl.sqrt(inner(ufl.dev(sigma_tr+chi1_old),ufl.dev(sigma_tr+chi1_old)))

				#residual
				r=inner(sigma,epsi(v))*dx-L
				rvec=fem.petsc.assemble_vector(fem.form(r))
				rvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
				fem.petsc.set_bc(rvec,bcs=[bc])
				res=rvec.norm()
				rvec.destroy()
				if msh.comm.rank==0:
					print("Residual norm: "+str(res),flush=True)
					residuals[m,ti]=res
				if res<eps: break

				i, j, k, l = ufl.indices(4)
				
				#subgradient
				S=ufl.conditional(ufl.eq(dl,0),ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l)),eval_S(sigma_tr,chi1_old,dl))

				tmp=ufl.dev(epsi(u))*2*mu+kappa*ufl.tr(epsi(u))*ufl.Identity(dim)
				tmp2=ufl.as_tensor(S[i,j,k,l]*tmp[k,l],(i,j))
				a= inner(tmp2,epsi(v)) * dx
				
				problem = LinearProblem(a, -r, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
				duh = problem.solve()
				
				uh.x.array[:]=uh.x.array[:]+duh.x.array[:]

			ep_f=0.5*(grad(uh)+ufl.transpose(grad(uh)))-ufl.dev(sigma)/(2*mu)-ufl.tr(sigma)*ufl.Identity(dim)/(kappa*dim**2)
			chi1_f=chi1_old-dl*k1*(ufl.dev(sigma_tr+chi1_old))/ufl.sqrt(inner(ufl.dev(sigma_tr+chi1_old),ufl.dev(sigma_tr+chi1_old)))
			chi2_f=chi2_old-dl*k2

			epexp=fem.Expression(ep_f, Vstr.element.interpolation_points())
			chi1exp=fem.Expression(chi1_f, Vstr.element.interpolation_points())
			chi2exp=fem.Expression(chi2_f, Vsca.element.interpolation_points())

			ep.interpolate(epexp)
			chi1.interpolate(chi1exp)
			chi2.interpolate(chi2exp)
			
			with io.XDMFFile(msh.comm, "uh"+str(ti+1)+".xdmf", "w") as file:
				file.write_mesh(msh)
				file.write_function(uh)

	#evalute deflection in the middle of narrow part
	cells=[]
	cell_candidates=geometry.compute_collisions_points(bb_tree,((2.5,0.25,0)))
	cells=geometry.compute_colliding_cells(msh,cell_candidates,((2.5,0.25,0)))
	if len(cells)>0:
		cells=cells[0]
		ydefl=uh.eval(((2.5,0.25,0)),cells)[1]
		print("Deflection: "+str(ydefl),flush=True)
		with open("defl_"+str(num_dofs_global)+".txt", 'w') as wfile:
			wfile.write("\n"+str(ydefl))	

	if msh.comm.rank==0:
		np.savetxt("res_"+str(num_dofs_global)+".csv",residuals,delimiter=",")
		print("DOFs: "+str(num_dofs_global))