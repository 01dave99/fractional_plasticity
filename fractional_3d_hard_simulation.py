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

dim=3
y0=10000
mu=55000
kappa=120000
k1=10000
k2=60000
alpha=0.4
I=10
tmax=15000
steps=200
eps=math.pow(10,-8)


#Specification of return mapping

def f(sigma,chi1,chi2):
	return(ufl.sqrt(inner(ufl.dev(sigma+chi1),ufl.dev(sigma+chi1)))+chi2-y0)

def df(sigma,chi1,chi2):
	df=ufl.dev(sigma+chi1)/(ufl.sqrt(ufl.inner(ufl.dev(sigma+chi1),ufl.dev(sigma+chi1))))
	return(df)

def C(epsi):
	return(2*mu*ufl.dev(epsi)+kappa*ufl.tr(epsi)*ufl.Identity(dim))

def epsi(u):
	return(0.5*(grad(u)+ufl.transpose(grad(u))))

def eval_gamma(sigma_tr,sigma,chi1,chi2,r):
	gamma=ufl.conditional(f(sigma_tr,chi1,chi2)<=0,0,(f(sigma_tr,chi1,chi2))/(ufl.inner(df(sigma_tr,chi1,chi2),C(r))+k1*ufl.inner(df(sigma_tr,chi1,chi2),df(sigma,chi1,chi2))+k2))
	return(gamma)

#trapezoidal rule of reformulation
def eval_r(sigma,chi1,chi2):
	i,j,k,l,n=ufl.indices(5)
	ones=ufl.as_tensor([[1 for p in range(dim)] for p in range(dim)])
	sig1=ufl.as_tensor(ones[i,j]*(sigma[k,l]+chi1[k,l])-I*ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l))
	dev1=ufl.as_tensor(sig1[i,j,k,l]-sig1[i,j,n,n]*ufl.Identity(dim)[k,l]/dim,(i,j,k,l))
	norm1=ufl.as_tensor([[ufl.sqrt(dev1[o,p,k,l]*dev1[o,p,k,l]) for o in range(dim)] for p in range(dim)])
	df1=ufl.as_tensor([[dev1[o,p,o,p]/norm1[o,p] for o in range(dim)] for p in range(dim)])
	ddf1=ufl.as_tensor([[((1-ufl.Identity(dim)[o,p]/dim)/norm1[o,p]-(dev1[o,p,o,p]**2)/norm1[o,p]**3) for o in range(dim)] for p in range(dim)])
	
	sig2=ufl.as_tensor(ones[i,j]*(sigma[k,l]+chi1[k,l])+I*ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l))
	dev2=ufl.as_tensor(sig2[i,j,k,l]-sig2[i,j,n,n]*ufl.Identity(dim)[k,l]/dim,(i,j,k,l))
	norm2=ufl.as_tensor([[ufl.sqrt(dev2[o,p,k,l]*dev2[o,p,k,l]) for o in range(dim)] for p in range(dim)])
	df2=ufl.as_tensor([[dev2[o,p,o,p]/norm2[o,p] for o in range(dim)] for p in range(dim)])
	ddf2=ufl.as_tensor([[((1-ufl.Identity(dim)[o,p]/dim)/norm2[o,p]-(dev2[o,p,o,p]**2)/norm2[o,p]**3) for o in range(dim)] for p in range(dim)])

	r1=I**(1-alpha)/scipy.special.gamma(2-alpha)*(df1+I/2*ddf1)
	r2=I**(1-alpha)/scipy.special.gamma(2-alpha)*(df2-I/2*ddf2)
	return(0.5*(r1+r2))

def eval_S(sigma_tr,sigma,chi1,chi2,r):
	i,j,k,l =ufl.indices(4)
	id4=ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l))
	S=id4-ufl.outer(C(r),df(sigma_tr,chi1,chi2))/(ufl.inner(df(sigma_tr,chi1,chi2),C(r))+k1*ufl.inner(df(sigma_tr,chi1,chi2),df(sigma,chi1,chi2))+k2)
	return(S)

#Finite Element specifications:

#mesh
for mshi in range(6):
	with io.XDMFFile(MPI.COMM_WORLD, "meshes/mesh3d_"+str(mshi)+".xdmf", "r") as xdmf:
	    msh = xdmf.read_mesh()
	#msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
	                            #points=((0.0, 0.0), (5.0, 1.0)), n=(64, 64),
	                            #cell_type=mesh.CellType.triangle)
	bb_tree = geometry.bb_tree(msh, msh.topology.dim)
	#function space:
	el=ufl.VectorElement("Lagrange",ufl.tetrahedron,1)
	elstr=ufl.TensorElement("DG",ufl.tetrahedron,0)
	elsca=ufl.FiniteElement("DG",ufl.tetrahedron,0)

	V=fem.functionspace(msh, el)
	num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
	if msh.comm.rank==0:
		print("Loaded mesh "+str(mshi)+" with "+str(num_dofs_global)+ " DOF.",flush=True)
	Vstr=fem.functionspace(msh,elstr)
	Vsca=fem.functionspace(msh,elsca)

	facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
	                                       marker=lambda x: np.isclose(x[0], 0.0))
	                                                                     
	dofs = fem.locate_dofs_topological(V=V, entity_dim=(msh.topology.dim - 1), entities=facets)
	bc = fem.dirichletbc(value=np.zeros(dim), dofs=dofs, V=V)

	facet_indices, facet_markers =[], []
	facets = mesh.locate_entities(msh, msh.topology.dim - 1,lambda x: np.isclose(x[0], 8))
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
	sigma_old=fem.Function(Vstr)
	ds=ufl.Measure("ds",domain=msh,subdomain_data=facet_tag)

	for ti in time:
		#b=ufl.as_vector([b1[ti],b2[ti]])
		t1=t1vec[ti]
		t2=t2vec[ti]
		t3=t2vec[ti]
		t=ufl.as_vector([t1,t2,t3])
		L=inner(t,v)* ds(1)


		if ti==0:
			for m in range(1000):
				if m==0:
					a=inner(C(epsi(u)),epsi(v))*dx
					
					problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
					uh=problem.solve()
				else:
					sigma_tr=C(epsi(uh))

					#stress return mapping
					dl=fem.Constant(msh,PETSc.ScalarType(0)) #first step is always elastic if timesteps are small enough
					sigma=sigma_tr

					#residual
					r=inner(sigma,epsi(v))*dx-L
					rvec=fem.petsc.assemble_vector(fem.form(r))
					rvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
					fem.petsc.set_bc(rvec,bcs=[bc])
					res=rvec.norm(norm_type=3)
					rvec.destroy()
					if msh.comm.rank==0:
						print("Residual norm: "+str(res),flush=True)
						residuals[m,ti]=res
					if res<eps: break
					
					i,j,k,l=ufl.indices(4)
					#subgradient
					S=ufl.conditional(ufl.eq(dl,0),ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l)),eval_S(sigma_tr,0*ufl.Identity(dim),0*ufl.Identity(dim),0,eval_r(0*ufl.Identity(dim),0*ufl.Identity(dim),0)))
					
					tmp=C(epsi(u))
					tmp2=ufl.as_tensor(S[i,j,k,l]*tmp[k,l],(i,j))
					a= inner(tmp2,epsi(v)) * dx
					
					problem = LinearProblem(a, -r, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
					duh = problem.solve()
					
					uh.x.array[:]=uh.x.array[:]+duh.x.array[:]

			ep_f=epsi(uh)-ufl.dev(sigma)/(2*mu)-ufl.tr(sigma)*ufl.Identity(dim)/(kappa*dim**2)
			chi1_f=-dl*k1*ufl.Identity(dim) #first step is always elastic if timestep is small enough.
			chi2_f=-dl*k2

			epexp=fem.Expression(ep_f, Vstr.element.interpolation_points())
			chi1exp=fem.Expression(chi1_f, Vstr.element.interpolation_points())
			chi2exp=fem.Expression(chi2_f, Vsca.element.interpolation_points())
			sigmaexp=fem.Expression(sigma, Vstr.element.interpolation_points())

			ep.interpolate(epexp)
			chi1.interpolate(chi1exp)
			chi2.interpolate(chi2exp)
			sigma_old.interpolate(sigmaexp)
			
			#with io.XDMFFile(msh.comm, "uh"+str(ti+1)+".xdmf", "w") as file:
			#	file.write_mesh(msh)
			#	file.write_function(uh[ti])
		else:

			ep_old=ep.copy()
			chi1_old=chi1.copy()
			chi2_old=chi2.copy()
			for m in range(1000):
				
				if m>20:
					if msh.comm.rank==0: 
						print("Stuck in loop...,skip.", flush=True)
					break
				sigma_tr=C(epsi(uh)-ep_old)

				#stress return mapping
				dl=eval_gamma(sigma_tr,sigma_old,chi1_old,chi2_old,eval_r(sigma_old,chi1_old,chi2_old))
				sigma=sigma_tr-dl*C(eval_r(sigma_old,chi1_old,chi2_old))

				#residual
				r=inner(sigma,epsi(v))*dx-L
				rvec=fem.petsc.assemble_vector(fem.form(r))
				rvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
				fem.petsc.set_bc(rvec,bcs=[bc])
				res=rvec.norm(norm_type=3)
				rvec.destroy()
				if msh.comm.rank==0:
					print("Residual norm: "+str(res),flush=True)
					residuals[m,ti]=res
				if res<eps: break
				
				i,j,k,l=ufl.indices(4)
				#subgradient
				S=ufl.conditional(ufl.eq(dl,0),ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l)),eval_S(sigma_tr,sigma_old,chi1_old,chi2_old,eval_r(sigma_old,chi1_old,chi2_old)))

				tmp=C(epsi(u))
				tmp2=ufl.as_tensor(S[i,j,k,l]*tmp[k,l],(i,j))
				a= inner(tmp2,epsi(v)) * dx
				
				problem = LinearProblem(a, -r, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
				duh = problem.solve()
				
				uh.x.array[:]=uh.x.array[:]+duh.x.array[:]

			ep_f=epsi(uh)-ufl.dev(sigma)/(2*mu)-ufl.tr(sigma)*ufl.Identity(dim)/(kappa*dim**2)
			chi1_f=chi1_old-dl*k1*df(sigma_old,chi1_old,chi2_old)
			chi2_f=chi2_old-dl*k2

			epexp=fem.Expression(ep_f, Vstr.element.interpolation_points())
			chi1exp=fem.Expression(chi1_f, Vstr.element.interpolation_points())
			chi2exp=fem.Expression(chi2_f, Vsca.element.interpolation_points())
			sigmaexp=fem.Expression(sigma, Vstr.element.interpolation_points())


			ep.interpolate(epexp)
			chi1.interpolate(chi1exp)
			chi2.interpolate(chi2exp)
			sigma_old.interpolate(sigmaexp)
			
			#with io.XDMFFile(msh.comm, "frac_uh"+str(ti+1)+".xdmf", "w") as file:
			#	file.write_mesh(msh)
			#	file.write_function(uh)

	if msh.comm.rank==0:
		np.savetxt("frac_res3d_"+str(num_dofs_global)+".csv",residuals,delimiter=",")
		print("DOFs: "+str(num_dofs_global))

	with io.XDMFFile(msh.comm, "frac_uh_"+str(num_dofs_global)+".xdmf", "w") as file:
				file.write_mesh(msh)
				file.write_function(uh)