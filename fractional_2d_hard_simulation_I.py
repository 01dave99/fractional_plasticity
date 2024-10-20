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

dim=2
y0=10000
mu=55000
kappa=55000
k1=110000
k2=110000
alpha=0.5
n_conv=10
I_count=0
Is=[np.array([[1,100],[100,1000]]),np.array([[5000,5000],[5000,5000]]),np.array([[200,100],[100,100]])]
tmax=15000
steps=200
eps=math.pow(10,-8)

with io.XDMFFile(MPI.COMM_WORLD, "meshes/mesh2d_11.xdmf", "r") as xdmf:
	    msh = xdmf.read_mesh()


for I in Is:
	#Specification of return mapping
	I_count+=1
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

	#convquad:implicit euler
	def eval_r(sigma,chi1,chi2):
		h=I/n_conv
		ws=ufl.as_tensor([scipy.special.gamma(k+1-alpha)/(scipy.special.gamma(1-alpha)*scipy.special.gamma(k+1)) for k in range(n_conv)])

		#i=j=0:
		hi=ufl.as_tensor(np.array([[h[0,0],0],[0,0]]))
		dfs=ufl.as_tensor([df(sigma-k*hi,chi1,chi2)[0,0]+df(sigma+k*hi,chi1,chi2)[0,0] for k in range(n_conv)])
		df1=ufl.inner(dfs,ws)*0.5

		#i=j=1:
		hi=ufl.as_tensor(np.array([[0,0],[0,h[1,1]]]))
		dfs=ufl.as_tensor([df(sigma-k*hi,chi1,chi2)[1,1]+df(sigma+k*hi,chi1,chi2)[1,1] for k in range(n_conv)])
		df2=ufl.inner(dfs,ws)*0.5

		#i=1 j=2:
		hi=ufl.as_tensor(np.array([[0,h[0,1]],[0,0]]))
		dfs=ufl.as_tensor([df(sigma-k*hi,chi1,chi2)[0,1]+df(sigma+k*hi,chi1,chi2)[0,1] for k in range(n_conv)])
		df12=ufl.inner(dfs,ws)*0.5

		r=ufl.as_tensor([[df1*h[0,0]**(1-alpha), df12*h[0,1]**(1-alpha)],[df12*h[0,1]**(1-alpha),df2*h[1,1]**(1-alpha)]])
		return(r/ufl.sqrt(ufl.inner(r,r)))

	def eval_S(sigma_tr,sigma,chi1,chi2,r):
		i,j,k,l =ufl.indices(4)
		id4=ufl.as_tensor(ufl.Identity(dim)[i,k]*ufl.Identity(dim)[j,l],(i,j,k,l))
		ddf=(id4-ufl.outer(ufl.Identity(dim),ufl.Identity(dim))/dim)/ufl.sqrt(ufl.inner(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1)))-ufl.outer(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1))/ufl.sqrt(ufl.inner(ufl.dev(sigma_tr+chi1),ufl.dev(sigma_tr+chi1)))**3
		ddf_prod=ufl.as_tensor(ddf[i,j,k,l]*(2*mu*r[k,l]+k1*df(sigma,chi1,chi2)[k,l]),(i,j))
		S2=ufl.outer(C(r),f(sigma_tr,chi1,chi2)*ddf_prod)/(ufl.inner(df(sigma_tr,chi1,chi2),C(r))+k1*ufl.inner(df(sigma_tr,chi1,chi2),df(sigma,chi1,chi2))+k2)**2
		S=id4-ufl.outer(C(r),df(sigma_tr,chi1,chi2))/(ufl.inner(df(sigma_tr,chi1,chi2),C(r))+k1*ufl.inner(df(sigma_tr,chi1,chi2),df(sigma,chi1,chi2))+k2)+S2
		return(S)

	#Finite Element specifications:

	
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
		print("I= "+str(I),flush=True)
		print("Loaded mesh with "+str(num_dofs_global)+" dofs.",flush=True)
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
	defl_right=np.zeros(steps)
	defl_mid=np.zeros(steps)

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
		t=ufl.as_vector([t1,t2])
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

		#evalute deflection in the middle of narrow part
		cells=[]
		cell_candidates=geometry.compute_collisions_points(bb_tree,((2.5,0.25,0)))
		cells=geometry.compute_colliding_cells(msh,cell_candidates,((2.5,0.25,0)))
		if len(cells)>0:
			cells=cells[0]
			ydefl=uh.eval(((2.5,0.25,0)),cells)[1]
			defl_mid[ti]=ydefl
		
		defl_mid[ti]=msh.comm.allreduce(defl_mid[ti],MPI.SUM)
		if msh.comm.rank==0:
			print("Deflection_mid: "+str(defl_mid[ti]),flush=True)

		#evaluate deflection at the right side of beam
		cells=[]
		cell_candidates=geometry.compute_collisions_points(bb_tree,((5,0.5,0)))
		cells=geometry.compute_colliding_cells(msh,cell_candidates,((5,0.5,0)))
		if len(cells)>0:
			cells=cells[0]
			ydefl=uh.eval(((5,0.5,0)),cells)[0]
			defl_right[ti]=ydefl
			
		defl_right[ti]=msh.comm.allreduce(defl_right[ti],MPI.SUM)
		if msh.comm.rank==0:
			print("Deflection_right: "+str(defl_right[ti]),flush=True)

	if msh.comm.rank==0:
		np.savetxt("results/frac_res_"+str(num_dofs_global)+"_I"+str(I_count)+".csv",residuals,delimiter=",")
		np.savetxt("results/defl_mid_"+str(num_dofs_global)+"_I"+str(I_count)+".csv",defl_mid,delimiter=",")
		np.savetxt("results/defl_right_"+str(num_dofs_global)+"_I"+str(I_count)+".csv",defl_right,delimiter=",")

	with io.XDMFFile(msh.comm, "results/frac_uh_final_"+str(num_dofs_global)+"_I"+str(I_count)+".xdmf", "w") as file:
				file.write_mesh(msh)
				file.write_function(uh)