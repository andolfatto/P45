from dolfin import *
import numpy as np
from subprocess import call
import datetime
from IPython import embed
import sys
from petsc4py import PETSc


# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


rint = 0.05668/2
rest = 0.12
res  = 1e-2
### impongo l'offset nel file input di gmsh, genero e converto la mesh
newline1='DefineConstant[ r1 = { %.4f, Path "Gmsh/Parameters"}];\n'  % rint
newline2='DefineConstant[ r2 = { %.4f, Path "Gmsh/Parameters"}];\n'  %rest
newline3='DefineConstant[ res= { %.4f, Path "Gmsh/Parameters"}];\n'  %res
with open('geom.geo') as fid:
	lines = fid.readlines()
lines[0] = lines[0].replace(lines[0], newline1)
lines[1] = lines[1].replace(lines[1], newline2)
lines[2] = lines[2].replace(lines[2], newline3)
with open('geom.geo', 'w') as fid2:
	for line in lines:
		fid2.write(line)



call(["gmsh", "geom.geo", "-2"])
call(["dolfin-convert", "geom.msh" , "mesh.xml"])
mesh = Mesh("mesh.xml")



L1 = FunctionSpace(mesh, 'Lagrange', 2)
L2 = FunctionSpace(mesh, 'Lagrange', 3)
L = MixedFunctionSpace([L2, L1, L1])







#MATERIAL DATA#
E = 9.1e+10#Pa
nu = 0.24
t = 2e-3#m
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
	return inner(as_vector([testw, testx, testy]), Lumped_K_tensor * as_vector([trialw, trialx, trialy])) * dp(1)
def MassMatrix(test, trial):
	"Computes mass matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	return inner(as_vector([testw, testx, testy]), M_tensor * as_vector([trialw, trialx, trialy])) * dx    
def LumpedMassMatrix(test, trial):
	"Computes lumped part of stiffness matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)   
	return inner(as_vector([testw, testx, testy]), Lumped_M_tensor * as_vector([trialw, trialx, trialy])) * dp(1)  
def forcew(f,test):
	"Computes force"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)   
	return inner(testw,f) * dp(1)  





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
Lumped_K_tensor = as_tensor([    [ -8.7391e+1 , 0.          , 0.           ],\
                                 [ 0.           , -3.82821e-7 , 0.           ],\
                                 [ 0.           , 0.          ,  -3.82821e-7 ]])

Lumped_M_tensor = as_tensor([    [ 4.1901e-3 , 0.        , 0.        ],\
                                 [ 0.        , 1.6301e-7 , 0.        ],\
                                 [ 0.        , 0.        , 1.6301e-7 ]])
class lumped(SubDomain):
	def inside(self, x, on_boundary):
		toll = 1e-5
		return  (near(x[0] , -2.3500000000000000e-02 , toll) and near(x[1] ,   1.0305000000000000e-01 , toll) ) or\
                        (near(x[0] , -5.2900000000000003e-02 , toll) and near(x[1] ,   9.1535000000000005e-02 , toll) ) or\
                        (near(x[0] , -9.5200000000000007e-02 , toll) and near(x[1] ,   4.5859999999999998e-02 , toll) ) or\
                        (near(x[0] , -1.0452000000000000e-01 , toll) and near(x[1] ,   1.5751999999999999e-02 , toll) ) or\
                        (near(x[0] , -1.0452000000000000e-01 , toll) and near(x[1] ,  -1.5800000000000002e-02 , toll) ) or\
                        (near(x[0] , -5.2900000000000003e-02 , toll) and near(x[1] ,  -9.1499999999999998e-02 , toll) ) or\
                        (near(x[0] ,  7.8989999999999998e-03 , toll) and near(x[1] ,  -1.0539999999999999e-01 , toll) ) or\
                        (near(x[0] ,  3.8614999999999997e-02 , toll) and near(x[1] ,  -9.8390000000000005e-02 , toll) ) or\
                        (near(x[0] , -7.7499999999999999e-02 , toll) and near(x[1] ,   7.1891999999999998e-02 , toll) ) or\
                        (near(x[0] ,  7.8989999999999998e-03 , toll) and near(x[1] ,   1.0539999999999999e-01 , toll) ) or\
                        (near(x[0] ,  6.5900000000000000e-02 , toll) and near(x[1] ,   8.2636000000000001e-02 , toll) ) or\
                        (near(x[0] ,  8.7330000000000005e-02 , toll) and near(x[1] ,   5.9540999999999997e-02 , toll) ) or\
                        (near(x[0] ,  1.0100000000000001e-01 , toll) and near(x[1] ,   3.1154000000000001e-02 , toll) ) or\
                        (near(x[0] ,  1.0100000000000001e-01 , toll) and near(x[1] ,  -3.1199999999999999e-02 , toll) ) or\
                        (near(x[0] ,  6.5900000000000000e-02 , toll) and near(x[1] ,  -8.2600000000000007e-02 , toll) ) or\
                        (near(x[0] , -7.7499999999999999e-02 , toll) and near(x[1] ,  -7.1900000000000006e-02 , toll) ) or\
                        (near(x[0] ,  6.8339999999999998e-02 , toll) and near(x[1] ,  -3.0400000000000000e-02 , toll) ) or\
                        (near(x[0] , -7.3200000000000001e-02 , toll) and near(x[1] ,   1.5553000000000001e-02 , toll) ) or\
                        (near(x[0] , -4.1300000000000003e-02 , toll) and near(x[1] ,   1.5016000000000000e-02 , toll) ) or\
                        (near(x[0] , -4.1300000000000003e-02 , toll) and near(x[1] ,  -1.4999999999999999e-02 , toll) ) or\
                        (near(x[0] , -6.0519999999999997e-02 , toll) and near(x[1] ,   4.3971000000000003e-02 , toll) ) or\
                        (near(x[0] ,  6.8339999999999998e-02 , toll) and near(x[1] ,   3.0426999999999999e-02 , toll) ) or\
                        (near(x[0] , -7.8200000000000006e-03 , toll) and near(x[1] ,   7.4397000000000005e-02 , toll) ) or\
                        (near(x[0] ,  2.3116999999999999e-02 , toll) and near(x[1] ,   7.1146000000000001e-02 , toll) ) or\
                        (near(x[0] ,  5.0056000000000003e-02 , toll) and near(x[1] ,   5.5592999999999997e-02 , toll) ) or\
                        (near(x[0] , -7.3200000000000001e-02 , toll) and near(x[1] ,  -1.5599999999999999e-02 , toll) ) or\
                        (near(x[0] , -7.8200000000000006e-03 , toll) and near(x[1] ,  -7.4399999999999994e-02 , toll) ) or\
                        (near(x[0] , -3.7400000000000003e-02 , toll) and near(x[1] ,  -6.4799999999999996e-02 , toll) ) or\
                        (near(x[0] ,  4.3903999999999999e-02 , toll) and near(x[1] ,   0.0000000000000000e+00 , toll) ) or\
                        (near(x[0] , -2.1999999999999999e-02 , toll) and near(x[1] ,   3.8022000000000000e-02 , toll) ) or\
                        (near(x[0] ,  7.6239999999999997e-03 , toll) and near(x[1] ,  -4.3200000000000002e-02 , toll) ) or\
                        (near(x[0] ,  7.6239999999999997e-03 , toll) and near(x[1] ,   4.3236999999999998e-02 , toll) ) or\
                        (near(x[0] ,  3.3632000000000002e-02 , toll) and near(x[1] ,   2.8221000000000000e-02 , toll) ) or\
                        (near(x[0] , -2.3500000000000000e-02 , toll) and near(x[1] ,  -1.0305000000000000e-01 , toll) ) or\
                        (near(x[0] ,  2.3116999999999999e-02 , toll) and near(x[1] ,  -7.1199999999999999e-02 , toll) ) or\
                        (near(x[0] , -9.5200000000000007e-02 , toll) and near(x[1] ,  -4.5859999999999998e-02 , toll) ) or\
                        (near(x[0] ,  1.0570000000000000e-01 , toll) and near(x[1] ,   0.0000000000000000e+00 , toll) ) or\
                        (near(x[0] ,  8.7330000000000005e-02 , toll) and near(x[1] ,  -5.9499999999999997e-02 , toll) ) or\
                        (near(x[0] , -6.0519999999999997e-02 , toll) and near(x[1] ,  -4.3999999999999997e-02 , toll) ) or\
                        (near(x[0] ,  5.0056000000000003e-02 , toll) and near(x[1] ,  -5.5599999999999997e-02 , toll) ) or\
                        (near(x[0] ,  7.4806999999999998e-02 , toll) and near(x[1] ,   0.0000000000000000e+00 , toll) ) or\
                        (near(x[0] , -3.7400000000000003e-02 , toll) and near(x[1] ,   6.4784999999999995e-02 , toll) ) or\
                        (near(x[0] ,  3.3632000000000002e-02 , toll) and near(x[1] ,  -2.8199999999999999e-02 , toll) ) or\
                        (near(x[0] , -2.1999999999999999e-02 , toll) and near(x[1] ,  -3.7999999999999999e-02 , toll) ) or\
                        (near(x[0] ,  3.8614999999999997e-02 , toll) and near(x[1] ,   9.8390000000000005e-02 , toll) )
                        
Lumped = lumped()
lumped_boundaries = MeshFunction("size_t", mesh, 0)
lumped_boundaries.set_all(0)
Lumped.mark(lumped_boundaries, 1)#set 1 to activate
dp = Measure('vertex')[lumped_boundaries]

#plot(lumped_boundaries, title = 'actuators')
#plot(w_boundaries, title = 'w constraint region')
#plot(theta_x_boundaries, title = 'theta_x constraint region')
#plot(theta_y_boundaries, title = 'theta_y constraint region')
#interactive()


#NORMAL MODES BUILDING#
test = TestFunction(L)
trial = TrialFunction(L)

#K#
K = assemble(StiffnessMatrix(test,trial)+LumpedStiffnessMatrix(test,trial))
#assemble(StiffnessMatrix(test,trial), tensor=K)
for bc in bcs: bc.apply(K)

#M#
lflag = True #set True to lump
if lflag:
	print 'Assembling lumped mass'
	mass_form = MassMatrix(test,trial)+LumpedMassMatrix(test,trial)
	I = Constant([1,0,0])
	mass_action_form = action(mass_form,I)
	M = assemble(mass_form)
	M.zero()
	M.set_diagonal(assemble(mass_action_form))
else:
	print 'Assembling consistent mass'
	M = assemble(MassMatrix(test,trial)+LumpedMassMatrix(test,trial))
#assemble(MassMatrix(test,trial), tensor=M)
for bc in bcs: bc.zero(M)

#C#
c_K = 1e-3
c_M = 1e-3
C = c_K*K + c_M * M

uvect = [0.]
uvect2 = [0.]

##EULERO
## Start time
#t0 = 0
#time = t0
#fexpr = Expression ("5",t = t0)#("5*sin(1e+3*t)",t=t0)#
#u0 = Function(L)
#u0.vector()[:] = 0
#udot0 = Function(L)
#udot0.vector()[:] = 0
## time step
#dt = 1e-4
## Define variational form of b
#rhs = forcew(fexpr,test)

#u1 = Function ( L )
#T = 500*dt
##p = plot(u0.sub(0),title="w", range_max = 0.005 , range_min = -0.005, window_width = 1920 , window_height = 1280)
##cycle

#left = M/(dt*dt)+C/dt+K
#LU = LUSolver(left)
#LU.parameters["reuse_factorization"] = True
#LU.parameters["symmetric"] = True
#LU.parameters["verbose"] = True
#while time <= T:
  #fexpr . t = time
  #b = assemble ( rhs )
  #bc . apply ( b )
  #right = b+(M/(dt*dt)+C/dt)*u0.vector()+M/dt*udot0.vector()
  #LU.solve(u1.vector(), right)
  #time += dt
  #print time
  #udot0 . assign( (u1-u0)/dt )
  #u0 . assign ( u1 )
  ##p.plot(u1.sub(0))
  #uvect2 = np.append(uvect2,max(u1.vector()))






#EULERO (1st step)  + MANTE

# Start time
t0 = 0
fexpr = Expression ("5",t = t0)#("5*sin(1e+3*t)",t=t0)#
u0 = Function(L)
u0.vector()[:] = 0
v0 = Function(L)
v0.vector()[:] = 0
a0 = Function(L)
a0.vector()[:] = 0
# time step
dt = 1e-4
# Define variational form of b
rhs = forcew(fexpr,test)

u1 = Function ( L )
v1 = Function ( L )
a1 = Function ( L )
time = dt
T = 10000*dt
q = plot(u0.sub(0),title="w", range_max = 0.005 , range_min = -0.005, window_width = 1920 , window_height = 1280)
#first iteration with Eulero
print time
fexpr . t = time
F = assemble ( rhs )
for bc in bcs: bc . apply ( F )
right = F+(M/(dt*dt)+C/dt)*u0.vector()+M/dt*v0.vector()
solve(M/(dt*dt)+C/dt+K, u1.vector(), right)
v1.assign ( (u1-u0)/dt )
a1.assign ( (v1-v0)/dt )
uvect = np.append(uvect,max(u1.vector()))
q.plot(u1.sub(0))
time += dt

#setting parameters
rho0 = .58
alpha = 1.
D = 2.*(1.+alpha)-(1.-rho0)**2
beta = alpha*((1.-rho0)**2*(2.+alpha)+2*(2.*rho0 -1.)*(1.+alpha))/D
delta = 0.5*alpha**2*(1.-rho0)**2/D
a_0 = 1-beta
a__1 = beta
b_1 = dt*(delta/alpha+alpha/2)
b_0 = dt*(beta/2+alpha/2-(1+alpha)*delta/alpha)
b__1 = dt*(beta/2+delta)
u2 = Function(L)
v2 = Function(L)
a2 = Function(L)

left = M + K*b_1*b_1 +C*b_1
LU = LUSolver(left)
LU.parameters["reuse_factorization"] = True
LU.parameters["symmetric"] = True
LU.parameters["verbose"] = True
#cycle
while time <= T:
  print 'mante'
  print time
  fexpr . t = time
  F = assemble ( rhs )
  bc . apply ( F )
  v_kk = a_0*v1.vector()+a__1*v0.vector()+b_0*a1.vector()+b__1*a0.vector()
  u_kk = a_0*u1.vector()+a__1*u0.vector()+(b_1*a_0+b_0)*v1.vector()+\
         (b_1*a__1+b__1)*v0.vector()+b_1*b_0*a1.vector()+b_1*b__1*a0.vector()
  right = F-K*u_kk-C*v_kk
  LU.solve(a2.vector(), right)
  time += dt
  v2.assign ( a_0*v1 + a__1*v0 + b_1*a2 + b_0*a1 + b__1*a0 )
  u2.assign ( a_0*u1 + a__1*u0 + b_1*v2 + b_0*v1 + b__1*v0 )


  q.plot(u2.sub(0))
  uvect = np.append(uvect,max(u2.vector()))
  #interactive()
  u0 . assign(u1)
  u1 . assign(u2)
  v0 . assign(v1)
  v1 . assign(v2)
  a0 . assign(a1)
  a1 . assign(a2)
  u2.vector()[:] = 0
  v2.vector()[:] = 0
  a2.vector()[:] = 0
 
with open('uplots.m','w') as fid:
	fid.write('uvect = [')
	for iter in range(np.size(uvect)):
	  fid.write('%f ' % uvect[iter])
	fid.write('];\n')
	fid.write('t = linspace(0,%f,%d);\n' % (1, np.size(uvect)))
	fid.write('plot(t,uvect)\ngrid on')
with open('uplotseu.m','w') as fid:
	fid.write('uvect2 = [')
	for iter in range(np.size(uvect2)):
	  fid.write('%f ' % uvect2[iter])
	fid.write('];\n')
	fid.write('t2 = linspace(0,%f,%d);\n' % (1, np.size(uvect2)))
	fid.write('plot(t2,uvect2,\'r\')\ngrid on')
 

  
  
  
  
  
  
interactive()
exit()
