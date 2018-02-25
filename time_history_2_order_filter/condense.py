from dolfin import *
import numpy as np
from subprocess import call
import datetime
from IPython import embed
import sys
from petsc4py import PETSc 
from actuators import *


# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


rint = 0.05668/2
rest = 0.12
res  = 3e-3
### impongo l'offset nel file input di gmsh, genero e converto la mesh
newline1='DefineConstant[ r1 = { %.4f, Path "Gmsh/Parameters"}];\n'  % rint
newline2='DefineConstant[ r2 = { %.4f, Path "Gmsh/Parameters"}];\n'  %rest
newline3='DefineConstant[ res= { %.4f, Path "Gmsh/Parameters"}];\n'  %res
with open('geom/geom.geo') as fid:
	lines = fid.readlines()
lines[0] = lines[0].replace(lines[0], newline1)
lines[1] = lines[1].replace(lines[1], newline2)
lines[2] = lines[2].replace(lines[2], newline3)
with open('geom/geom.geo', 'w') as fid2:
	for line in lines:
		fid2.write(line)


act_coords = np.load("data/act_coords.npy")
call(["gmsh", "geom/geom.geo", "-2"])
call(["dolfin-convert", "geom/geom.msh" , "geom/mesh.xml"])
mesh = Mesh("geom/mesh.xml")

L1 = FunctionSpace(mesh, 'Lagrange', 2)
L2 = FunctionSpace(mesh, 'Lagrange', 3)
L = MixedFunctionSpace([L2, L1, L1])


#MATERIAL DATA#
E = 9.1e+10#Pa
nu = 0.24
t = 0.00161
rho = 2530##Kg/m^3



t3 = t*t*t
G = E/2/(1+nu)
nu2 = nu*nu
D_tensor = E*t3/12/(1-nu2)*as_tensor([    [   1  ,  nu  ,    0     ] ,\
                                          [  nu  ,   1   ,    0    ] ,\
                                          [   0  ,   0  , (1-nu)/2 ] ])

M_tensor = rho*as_tensor([   [   t    ,   0      ,    0  ] ,\
                             [   0    ,   t3/12  ,    0  ] ,\
                             [   0    ,   0      , t3/12 ] ])


def eps_fl(thx,thy):
	"Computes flexional deformation vector according to mindlin plate"
	mx = as_tensor([[0 , 0],\
			[0 ,-1],\
			[1 , 0]])
	my = as_tensor([[1 , 0],\
			[0 , 0],\
			[0 ,-1]])
	return mx*nabla_grad(thx)+my*nabla_grad(thy)
      
def eps_sh(w,thx,thy):
	"Computes shear deformation vector according to mindlin plate"
	mx = as_vector([0 , -1])
	
	my = as_vector([1 , 0])
	return mx*thx+my*thy+nabla_grad(w)

def StiffnessMatrix(test, trial):
	"Computes stiffness matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	return (inner(eps_fl(testx , testy) , D_tensor * eps_fl(trialx, trialy))+\
	        inner(eps_sh(testw , testx , testy) , eps_sh(trialw , trialx , trialy) ) * G * t  ) *dx

def LumpedStiffnessMatrix(test, trial):
	"Computes lumped part of stiffness matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial) 
	output = inner(as_vector([testw, testx, testy]),\
	         Lumped_K_tensor * as_vector([trialw, trialx, trialy]))
	return output * dp(1)
def MassMatrix(test, trial):
	"Computes mass matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	return inner(as_vector([testw, testx, testy]), M_tensor * as_vector([trialw, trialx, trialy])) * dx    
def LumpedMassMatrix(test, trial):
	"Computes lumped part of stiffness matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	output = inner(as_vector([testw, testx, testy]),\
	         Lumped_M_tensor * as_vector([trialw, trialx, trialy]))
	return output * dp(1)
def forcew(f,test,DX):
	"Computes force"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	output = inner(testw,f)
	return output * DX





#DIRICHLET BCS#
#w#
w_val = Constant(0.)
class w_constraint(SubDomain):
	def inside(self, x, on_boundary):
		return  on_boundary and x[0]*x[0]+x[1]*x[1]<rint*rint*1.1
w_Constraint = w_constraint()
w_boundaries = MeshFunction("size_t", mesh, 1)
w_boundaries.set_all(0)
w_Constraint.mark(w_boundaries, 1)#set 1 to activate, , 0 to deactivate
w_bc = DirichletBC(L.sub(0), w_val, w_boundaries,1)
#theta_x#
theta_x_val = Constant(0.)
class theta_x_constraint(SubDomain):
	def inside(self, x, on_boundary):
		return  on_boundary and x[0]*x[0]+x[1]*x[1]<rint*rint*1.1
theta_x_Constraint = theta_x_constraint()
theta_x_boundaries = MeshFunction("size_t", mesh, 1)
theta_x_boundaries.set_all(0)
theta_x_Constraint.mark(theta_x_boundaries, 0)#set 1 to activate, 0 to deactivate
theta_x_bc = DirichletBC(L.sub(1), theta_x_val, theta_x_boundaries,1)
#theta_y#
theta_y_val = Constant(0.)
class theta_y_constraint(SubDomain):
	def inside(self, x, on_boundary):
		return  on_boundary and x[0]*x[0]+x[1]*x[1]<rint*rint*1.1
theta_y_Constraint = theta_y_constraint()
theta_y_boundaries = MeshFunction("size_t", mesh, 1)
theta_y_boundaries.set_all(0)
theta_y_Constraint.mark(theta_y_boundaries, 0)#set 1 to activate,q 0 to deactivate
theta_y_bc = DirichletBC(L.sub(2), theta_y_val, theta_y_boundaries,1)

bcs = [ w_bc , theta_x_bc , theta_y_bc ]



#LUMPED PARAMETERS#
Lumped_K_tensor = as_tensor([    [ -87.4 , 0.          , 0.           ],\
                                 [ 0.           , 7e-5 , 0.           ],\
                                 [ 0.           , 0.          ,  7e-5 ]])

Lumped_M_tensor = as_tensor([    [ 4.3201e-3 , 0.        , 0.        ],\
                                 [ 0.        , 1.6418e-7 , 0.        ],\
                                 [ 0.        , 0.        , 1.6418e-7 ]])

act = ActList(L, act_coords)
n_act = np.shape(act_coords)[0]
lumped_boundaries = MeshFunction("size_t", mesh, 0)
lumped_boundaries.set_all(0)

act.mark_all(lumped_boundaries,1)
dp = act.dp_all()
#act.plot_coords()
#plot(lumped_boundaries, title = 'actuators')
#plot(w_boundaries, title = 'w constraint region')
#plot(theta_x_boundaries, title = 'theta_x constraint region')
#plot(theta_y_boundaries, title = 'theta_y constraint region')
#interactive()
#exit()

#NORMAL MODES BUILDING#
test = TestFunction(L)
trial = TrialFunction(L)

#K#
print "Assembling K"
K = assemble(StiffnessMatrix(test,trial)+LumpedStiffnessMatrix(test,trial))
#assemble(StiffnessMatrix(test,trial), tensor=K)
#for bc in bcs: bc.apply(K)

#M#
print "Assembling M"
lflag = True #set True to lump
if lflag:
	#print 'Assembling lumped mass'
	mass_form = MassMatrix(test,trial)+LumpedMassMatrix(test,trial)
	I = Constant([1,0,0])
	mass_action_form = action(mass_form,I)
	M = assemble(mass_form)
	M.zero()
	M.set_diagonal(assemble(mass_action_form))
else:
	#print 'Assembling consistent mass'
	M = assemble(MassMatrix(test,trial)+LumpedMassMatrix(test,trial))
#assemble(MassMatrix(test,trial), tensor=M)
#for bc in bcs: bc.zero(M)

#C#
print "Assembling C"
c_K = 0
c_M = 1e-3
C = c_K*K + c_M * M
       
#M ridotta
print "Condensing M"
fakefun = Function(FunctionSpace(mesh,'DG',0))
fakefun.vector()[:] = 1
mtot = assemble(fakefun*rho*t*dx)+4.3201e-3*n_act
vect = mtot/n_act*np.ones([n_act])
Mred = np.zeros(n_act*n_act)
Mred = Mred.reshape(n_act,n_act)
np.fill_diagonal(Mred, vect)
np.save("data/Mred",Mred)

##K ridotta
print "Condensing K"
Kred = np.array([])
K2 = K.copy()
for i in range(n_act):
    bc = DirichletBC(L.sub(0), 0.0, act.pinpoint(i), 'pointwise')
    bc.apply(K2)
LU = LUSolver(K2)
LU.parameters["reuse_factorization"] = True
LU.parameters["symmetric"] = True
LU.parameters["verbose"] = True
utemp = Function(L)
R = Function(L)
f = assemble(forcew(Constant(0),test,dx()))
for i in range(n_act):
  print 'colonna %d' % i
  f[:] = 0
  utemp.vector()[:] = 0
  bc = DirichletBC(L.sub(0), 1, act.pinpoint(i), 'pointwise')
  bc.apply(f)
  LU.solve(utemp.vector(), f)
  #plot(utemp.sub(0))
  R.vector()[:] = K*utemp.vector()[:]
  #plot(R.sub(0))
  #act.plot_displacement(utemp)
  #interactive()
  Rv = R.sub(0).compute_vertex_values()
  Kred = np.append(Kred, Rv[act.vertex_id])  

Kred = Kred.reshape(n_act,n_act)
Kred = np.transpose(Kred)

np.save("data/Kred",Kred)




#K ridotta  2
print "Condensing K with capacitance reading error"
Forcevect = np.array([])
Dispvect = np.array([])
K2 = K.copy()
for i in range(n_act):
    bc = DirichletBC(L.sub(0), 0.0, act.pinpoint(i), 'pointwise')
    bc.apply(K2)
LU = LUSolver(K2)
LU.parameters["reuse_factorization"] = True
LU.parameters["symmetric"] = True
LU.parameters["verbose"] = True
utemp = Function(L)
R = Function(L)
f = assemble(forcew(Constant(0),test,dx()))
for i in range(n_act):
  print 'colonna %d' % i
  f[:] = 0
  utemp.vector()[:] = 0
  bc = DirichletBC(L.sub(0), 1.e-6, act.pinpoint(i), 'pointwise')
  bc.apply(f)
  LU.solve(utemp.vector(), f)
  #plot(utemp.sub(0))
  R.vector()[:] = K*utemp.vector()[:]
  #plot(R.sub(0))
  #act.plot_displacement(utemp)
  #interactive()
  
  Rv = R.sub(0).compute_vertex_values()
  Forcevect = np.append(Forcevect, Rv[act.vertex_id])
  
  Uv = act.disp2disp_all_clean(utemp)
  Dispvect = np.append(Dispvect, Uv) 
  
Forcevect = Forcevect.reshape(n_act,n_act)
Forcevect = np.transpose(Forcevect)

Dispvect = Dispvect.reshape(n_act,n_act)
Dispvect = np.transpose(Dispvect)

Dispinv = np.linalg.inv(Dispvect)

Kred2 = Forcevect.dot(Dispinv)

np.save("data/Kred_with_cond",Kred2)

embed()