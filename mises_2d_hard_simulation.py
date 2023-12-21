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
steps=200
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
for meshi in range(6,8):
	#mesh
	print("Loading mesh "+str(meshi)+".\n ########################")

	with io.XDMFFile(MPI.COMM_WORLD, "mesh2d_"+str(meshi)+".xdmf", "r") as xdmf:
	    msh = xdmf.read_mesh()
	#msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
	                            #points=((0.0, 0.0), (5.0, 1.0)), n=(64, 64),
	                            #cell_type=mesh.CellType.triangle)
	
	#function space:
	el=ufl.VectorElement("Lagrange",ufl.triangle,1)
	elstr=ufl.TensorElement("DG",ufl.triangle,0)
	elsca=ufl.FiniteElement("DG",ufl.triangle,0)

	V=fem.functionspace(msh, el)
	num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
	Vstr=fem.functionspace(msh,elstr)
	Vsca=fem.functionspace(msh,elsca)

	facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
	                                       marker=lambda x: np.isclose(x[0], 0.0))
	                                                                     
	dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
	bc = fem.dirichletbc(value=np.zeros(dim), dofs=dofs, V=V)

	facets = mesh.locate_entities(msh, msh.topology.dim - 1,lambda x: np.isclose(x[0], 5))
	facet_indices = np.hstack(facets).astype(np.int32)
	facet_markers=np.full_like(facets, 1)
	facet_markers = np.hstack(facet_markers).astype(np.int32)
	sorted_facets = np.argsort(facet_indices)
	facet_tag = mesh.meshtags(msh, msh.topology.dim - 1, facet_indices[sorted_facets], facet_markers[sorted_facets])

	#functions:
	x = ufl.SpatialCoordinate(msh)

	#b1=[0,0,0]
	#b2=[-20000,-25000,-30000]


	t1vec=np.zeros(steps)
	for i in range(steps):
		t1vec[i]=-abs(2*tmax*(i+1)/steps-tmax)+tmax
	t1vec[steps-1]=0.1
	t2vec=np.zeros(steps)
	time=range(steps)

	residuals=np.zeros((1000,steps))

	u=ufl.TrialFunction(V)
	v=ufl.TestFunction(V)

	uh=list()
	sigma_tr=list()
	sigma=list()
	ep=list()
	chi1=list()
	chi2=list()
	dl=list()
	sigma_f=list()
	ep_f=list()
	chi1_f=list()
	chi2_f=list()
	dl_f=list()
	base_f=fem.Function(Vstr)
	base_f_sca=fem.Function(Vsca)
	ds=ufl.Measure("ds",domain=msh,subdomain_data=facet_tag)

	for ti in time:
		#b=ufl.as_vector([b1[ti],b2[ti]])
		t1=t1vec[ti]
		t2=t2vec[ti]
		t=ufl.as_vector([t1,t2])
		L=inner(t,v)* ds(1)

	#initialization
		uh.append(ufl.as_vector([0,0]))
		sigma_tr.append(0*ufl.Identity(dim))
		sigma.append(0*ufl.Identity(dim))
		ep.append(0*ufl.Identity(dim))
		chi1.append(0*ufl.Identity(dim))
		chi2.append(0)
		dl.append(0)

		sigma_f.append(base_f.copy())
		ep_f.append(base_f.copy())
		chi1_f.append(base_f.copy())
		chi2_f.append(base_f_sca.copy())
		dl_f.append(base_f_sca.copy())

		if ti==0:
			for m in range(1000):
				if m==0:
					a=inner(ufl.dev(epsi(u))*2*mu+kappa*ufl.tr(epsi(u))*ufl.Identity(dim),epsi(v))*dx
					problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
					uh[ti]=problem.solve()
				else:
					sigma_tr[ti]=ufl.dev(epsi(uh[ti]))*2*mu+kappa*ufl.tr(epsi(uh[ti]))*ufl.Identity(dim)

					#stress return mapping
					dl[ti]=eval_gamma(sigma_tr[ti],0*ufl.Identity(dim),0)
					sigma[ti]=sigma_tr[ti]-2*mu*dl[ti]*(ufl.dev(sigma_tr[ti]))/ufl.sqrt(inner(ufl.dev(sigma_tr[ti]),ufl.dev(sigma_tr[ti])))

					#residual
					r=inner(sigma[ti],epsi(v))*dx-L
					rvec=fem.petsc.assemble_vector(fem.form(r))
					fem.petsc.set_bc(rvec,bcs=[bc])
					print("Residual norm: "+str(rvec.norm()))
					residuals[m,ti]=rvec.norm()
					if rvec.norm()<eps: break

					i, j, k, l = ufl.indices(4)
					
					#subgradient
					S=ufl.conditional(ufl.eq(dl[ti],0),ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l)),eval_S(sigma_tr[ti],0*ufl.Identity(dim),dl[ti]))
					
					tmp=ufl.dev(epsi(u))*2*mu+kappa*ufl.tr(epsi(u))*ufl.Identity(dim)
					tmp2=ufl.as_tensor(S[i,j,k,l]*tmp[k,l],(i,j))
					a= inner(tmp2,epsi(v)) * dx
					
					problem = LinearProblem(a, -r, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
					duh = problem.solve()
					
					uh[ti].x.array[:]=uh[ti].x.array[:]+duh.x.array[:]

			ep[ti]=0.5*(grad(uh[ti])+ufl.transpose(grad(uh[ti])))-ufl.dev(sigma[ti])/(2*mu)-ufl.tr(sigma[ti])*ufl.Identity(dim)/(kappa*dim**2)
			chi1[ti]=-dl[ti]*k1*(ufl.dev(sigma_tr[ti]))/ufl.sqrt(inner(ufl.dev(sigma_tr[ti]),ufl.dev(sigma_tr[ti])))
			chi2[ti]=-dl[ti]*k2

			sigmaexp=fem.Expression(sigma[ti], Vstr.element.interpolation_points())
			epexp=fem.Expression(ep[ti], Vstr.element.interpolation_points())
			chi1exp=fem.Expression(chi1[ti], Vstr.element.interpolation_points())
			chi2exp=fem.Expression(chi2[ti], Vsca.element.interpolation_points())
			dlexp=fem.Expression(dl[ti], Vsca.element.interpolation_points())

			sigma_f[ti].interpolate(sigmaexp)
			ep_f[ti].interpolate(epexp)
			chi1_f[ti].interpolate(chi1exp)
			chi2_f[ti].interpolate(chi2exp)
			dl_f[ti].interpolate(dlexp)
			
			#with io.XDMFFile(msh.comm, "uh"+str(ti+1)+".xdmf", "w") as file:
			#	file.write_mesh(msh)
			#	file.write_function(uh[ti])
		else:

			uh[ti]=uh[ti-1].copy()
			for m in range(1000):
			
				sigma_tr[ti]=ufl.dev(epsi(uh[ti])-ep_f[ti-1])*2*mu+kappa*ufl.tr(epsi(uh[ti])-ep_f[ti-1])*ufl.Identity(dim)

				#stress return mapping
				dl[ti]=eval_gamma(sigma_tr[ti],chi1_f[ti-1],chi2_f[ti-1])
				sigma[ti]=sigma_tr[ti]-2*mu*dl[ti]*(ufl.dev(sigma_tr[ti]+chi1_f[ti-1]))/ufl.sqrt(inner(ufl.dev(sigma_tr[ti]+chi1_f[ti-1]),ufl.dev(sigma_tr[ti]+chi1_f[ti-1])))

				#residual
				r=inner(sigma[ti],epsi(v))*dx-L
				rvec=fem.petsc.assemble_vector(fem.form(r))
				fem.petsc.set_bc(rvec,bcs=[bc])
				print("Residual norm: "+str(rvec.norm()))
				residuals[m,ti]=rvec.norm()
				if rvec.norm()<eps: break

				i, j, k, l = ufl.indices(4)
				
				#subgradient
				S=ufl.conditional(ufl.eq(dl[ti],0),ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l)),eval_S(sigma_tr[ti],chi1_f[ti-1],dl[ti]))

				tmp=ufl.dev(epsi(u))*2*mu+kappa*ufl.tr(epsi(u))*ufl.Identity(dim)
				tmp2=ufl.as_tensor(S[i,j,k,l]*tmp[k,l],(i,j))
				a= inner(tmp2,epsi(v)) * dx
				
				problem = LinearProblem(a, -r, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
				duh = problem.solve()
				
				uh[ti].x.array[:]=uh[ti].x.array[:]+duh.x.array[:]

			ep[ti]=0.5*(grad(uh[ti])+ufl.transpose(grad(uh[ti])))-ufl.dev(sigma[ti])/(2*mu)-ufl.tr(sigma[ti])*ufl.Identity(dim)/(kappa*dim**2)
			chi1[ti]=chi1_f[ti-1]-dl[ti]*k1*(ufl.dev(sigma_tr[ti]+chi1_f[ti-1]))/ufl.sqrt(inner(ufl.dev(sigma_tr[ti]+chi1_f[ti-1]),ufl.dev(sigma_tr[ti]+chi1_f[ti-1])))
			chi2[ti]=chi2_f[ti-1]-dl[ti]*k2

			sigmaexp=fem.Expression(sigma[ti], Vstr.element.interpolation_points())
			epexp=fem.Expression(ep[ti], Vstr.element.interpolation_points())
			chi1exp=fem.Expression(chi1[ti], Vstr.element.interpolation_points())
			chi2exp=fem.Expression(chi2[ti], Vsca.element.interpolation_points())
			dlexp=fem.Expression(dl[ti], Vsca.element.interpolation_points())

			sigma_f[ti].interpolate(sigmaexp)
			ep_f[ti].interpolate(epexp)
			chi1_f[ti].interpolate(chi1exp)
			chi2_f[ti].interpolate(chi2exp)
			dl_f[ti].interpolate(dlexp)
			
			#with io.XDMFFile(msh.comm, "uh"+str(ti+1)+".xdmf", "w") as file:
			#	file.write_mesh(msh)
			#	file.write_function(uh[ti])

	#evalute deflection in the middle of narrow part
	bb_tree = geometry.bb_tree(msh, msh.topology.dim)
	cell_candidates=geometry.compute_collisions_points(bb_tree,((2.5,0.25,0)))
	cells=geometry.compute_colliding_cells(msh,cell_candidates,((2.5,0.25,0)))
	if len(cells)>1:
		cells=cells[0]
	ydefl=uh[steps-1].eval(((2.5,0.25,0)),cells)[1]	

	np.savetxt("res_"+str(num_dofs_global)+".csv",residuals,delimiter=",")
	print("DOFs: "+str(num_dofs_global))
	print("Deflection: "+str(ydefl))
	with open("defl_"+str(num_dofs_global)+".txt", 'w') as wfile:
	    wfile.write("\n"+str(ydefl))