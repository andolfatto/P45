#export OMP_NUM_THREADS=1
from dolfin import *
import numpy as np
from subprocess import call
import datetime
from IPython import embed
import sys
from petsc4py import PETSc 
from actuators import *
from feedforward import *
from feedback import *
from integrators import *
from mpl_toolkits.mplot3d import Axes3D                                          
import matplotlib.pyplot as plt   

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
with open('geom/geom.geo') as fid:
	lines = fid.readlines()
lines[0] = lines[0].replace(lines[0], newline1)
lines[1] = lines[1].replace(lines[1], newline2)
lines[2] = lines[2].replace(lines[2], newline3)
with open('geom/geom.geo', 'w') as fid2:
	for line in lines:
		fid2.write(line)



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
G = E/2./(1.+nu)
nu2 = nu*nu
D_tensor = E*t3/12./(1.-nu2)*as_tensor([    [   1.  ,  nu  ,    0     ] ,\
                                          [  nu  ,   1.   ,    0    ] ,\
                                          [   0  ,   0  , (1.-nu)/2. ] ])

M_tensor = rho*as_tensor([   [   t    ,   0      ,    0  ] ,\
                             [   0    ,   t3/12.  ,    0  ] ,\
                             [   0    ,   0      , t3/12. ] ])


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
def inverseMassMatrix(test, trial):
	"Computes mass matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	return inner(as_vector([testw, testx, testy]), M_inv * as_vector([trialw, trialx, trialy])) * dx  
def LumpedMassMatrix(test, trial):
	"Computes lumped part of stiffness matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	output = inner(as_vector([testw, testx, testy]),\
	         Lumped_M_tensor * as_vector([trialw, trialx, trialy]))
	return output * dp(1)
def inverseLumpedMassMatrix(test, trial):
	"Computes lumped part of stiffness matrix according with mindlin plate's theory"
	(testw,testx,testy) = split(test)
	(trialw,trialx,trialy) = split(trial)
	output = inner(as_vector([testw, testx, testy]),\
	         Lumped_M_inv * as_vector([trialw, trialx, trialy]))
	return output * dp(1)
def forcew(f,test,DX):
	"Computes force"
	(testw,testx,testy) = split(test)
	output = inner(testw,f)
	return output * DX
def forceall(f,test,DX):
	"Computes force"
	(testw,testx,testy) = split(test)
	output = inner(testw,f[0])+inner(testx,f[1])+inner(testy,f[2])
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

bcs = [ w_bc]# , theta_x_bc , theta_y_bc ]



#LUMPED PARAMETERS#
Lumped_K_tensor = as_tensor([    [ -87.4 , 0.          , 0.           ],\
                                 [ 0.           , 7e-5 , 0.           ],\
                                 [ 0.           , 0.          ,  7e-5 ]])

Lumped_M_tensor = as_tensor([    [ 4.3201e-3 , 0.        , 0.        ],\
                                 [ 0.        , 1.6418e-7 , 0.        ],\
                                 [ 0.        , 0.        , 1.6418e-7 ]])


Lumped_M_inv = as_tensor([    [ 1./4.3201e-3 , 0.        , 0.        ],\
                                 [ 0.        , 1./1.6418e-7 , 0.        ],\
                                 [ 0.        , 0.        , 1./1.6418e-7 ]])
M_inv = 11./30.*1./rho*as_tensor([   [   1./t    ,   0      ,    0  ] ,\
                             [   0    ,   1./(t3/12.)  ,    0  ] ,\
                             [   0    ,   0      , 1./(t3/12.) ] ])

act_coords = np.load("data/act_coords.npy")
act = ActList(L, act_coords)
n_act = np.shape(act_coords)[0]
lumped_boundaries = MeshFunction("size_t", mesh, 0)
lumped_boundaries.set_all(0)

#act.mark(lumped_boundaries,20,0)
#act.mark(lumped_boundaries,44,44)

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

##M#
print "Assembling M"
lflag = False #set True to lump
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
##assemble(MassMatrix(test,trial), tensor=M)
##for bc in bcs: bc.zero(M)

############################################
#nuova routine per M diagonale senza zeri

#Minv = assemble(inverseMassMatrix(test,trial)+inverseLumpedMassMatrix(test,trial))   

#mdiag = Function(L)
#Minv.get_diagonal(mdiag.vector())
#M = assemble(MassMatrix(test,trial)+LumpedMassMatrix(test,trial))   
#pippo = Function(L)
#M.get_diagonal(pippo.vector())
#M.zero()
#M.set_diagonal(pippo.vector())
#embed()
#C#
print "Assembling C"

###calcolo alfa e beta in funzione di 2 punti nel grafico risolvendo il sistema 2*csi = alfa/w +beta*w
csi1 = .5
csi2 = 1e-3
fre1 = 1
fre2 = 1e3
Z = np.array([1/(2*np.pi*fre1), (2*np.pi*fre1), 1/(2*np.pi*fre2), (2*np.pi*fre2)  ]).reshape(2,2)
Q = np.array([2*csi1, 2*csi2])
alfabeta = np.linalg.solve(Z,Q)
print alfabeta[0]
print alfabeta[1]

#alfabeta = [6.28, 0]
C = alfabeta[0]* M + alfabeta[1]*K






# Start time
t0 = 0
#dt
dt = 1./80000.
ni = 5
idt = dt/ni
# Period of 1 step
T = 1e-3

                                                                                   
                                                                                      
steps = np.loadtxt("data/story_P45.bin")
udes =1e-7*np.ones(n_act)                                                                                          
                                                                                      
                                                                                      
flag = True   # set True to insert the reading error simulation (in ff & fb)          


                                                                                
#random offset for actuators
gamma = 0
mean_offset = 0.5e-3
mux = mean_offset*np.cos(gamma)
muy = mean_offset*np.sin(gamma)
sigma = 0.2e-3
offx = np.zeros(n_act)
offy = np.zeros(n_act)
#for i in range(n_act):
  #offx[i] = np.random.normal(mux,sigma)     ####SONO SPENTI ORA####
  #offy[i] = np.random.normal(muy,sigma)
  
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(offx, offy)  
#plt.grid()
#plt.show()

#exit()


#FEEDFORWARD                                                                          
ff = FeedForward(udes, udes, T, act, flag)                                                        
ff.alfa = alfabeta[0]                                                                        
ff.beta = alfabeta[1]                                                                         
ff.recompute_C()                                                                      

#FEEDBACK
gp = 20000
gd = 15
fb = FeedBack(act,dt,gp*np.ones(n_act),gd*np.ones(n_act),flag)



fcost = 0
f = forcew(Constant(0),test,dx)

#######################ENABLE THESE 2 LINES TO ACTIVATE GRAVITY##############################
#fcost = 0.2507-9.81*ff.M[0,0]
#f = forcew(Constant(9.81*rho*t),test,dx)+forcew(Constant(-0.2507+9.81*4.3201e-3),test, dp(1))
#############################################################################################


fcostvector = np.zeros(n_act)
fcostvector[:] = fcost


F = assemble(f)
fvect = assemble(f)
fvect[:] = F[:]

Fold = assemble(f)

eu = Eulero(L, M, C, K, idt)
mnt = Mantegazza(L, M, C, K, idt, 1)
sens = SensorFilter(40000, n_act, idt, 3.188993986668676e-11)
filt = SensorFilter(25000, n_act, idt, 0)
tv = np.array([0])
my_filter_for_vel = SensorFilter(25000, n_act, idt, 0)

Res = 2.428
Ind = 3.6829e-4
Ki = 2.025
Current = CurrentFilter(Res, Ind, n_act, idt)
itemp = np.zeros(n_act)
Rp = 20*Res
iobserver = np.zeros(n_act)##
vobserver = np.zeros(n_act)##
iobserver_old = np.zeros(n_act)##
vobserver_old = np.zeros(n_act)##
iobserver_vect = np.zeros(n_act).reshape(n_act,1)
vobserver_vect = np.zeros(n_act).reshape(n_act,1)
voltemp = np.zeros(n_act)
tracevoltage = np.zeros(n_act).reshape(n_act,1)
tracecurrent = np.zeros(n_act).reshape(n_act,1)


trace = np.zeros(n_act).reshape(n_act,1)
tracetrue = np.zeros(n_act).reshape(n_act,1)
tracevel = np.zeros(n_act).reshape(n_act,1)
traceveltrue = np.zeros(n_act).reshape(n_act,1)
itrace = np.array([0.])
ireq = np.array([0.])
itv = np.array([0.])
ref = np.zeros(n_act).reshape(n_act,1)
refvel = np.zeros(n_act).reshape(n_act,1)
curr = np.zeros(n_act)

utemp = np.zeros(n_act)
utemp2 = np.zeros(n_act)

w_k1 = np.zeros(n_act)
w_k = np.zeros(n_act)
v_k = np.zeros(n_act)


oxystoring = np.zeros(n_act).reshape(n_act,1)
oxy = np.zeros(n_act)
foxystoring = np.zeros(n_act).reshape(n_act,1)
foxy = np.zeros(n_act)
oxycount =0.


#####Parametri per filtro Kalman
Ll = np.array([0.4124, 0.7666])
a = .75
ufilt = np.zeros(n_act)##la u filtrata (riga 1 per il mante)
urec = np.zeros(n_act)## la u ricostruita (riga 2 per il mante)
ufiltold = np.zeros(n_act)##
urecold = np.zeros(n_act)##
#####

call(["rm", "results/utemp.dat"])
call(["rm", "results/udtemp.dat"])
call(["rm", "results/utruetemp.dat"])
call(["rm", "results/udtruetemp.dat"])
call(["rm", "results/timetemp.dat"])

#############################
#for new force update methods
dpact = act.dp()
Cons0 = []
Cons1 = []
Cons2 = []
Cons0.append(Constant(0.))
Cons1.append(Constant(0.))
Cons2.append(Constant(0.))
Fform = forceall([Cons0[0],Cons1[0],Cons2[0]],test,dpact(1))
for i in range(1,n_act):	
  Cons0.append(Constant(0.))
  Cons1.append(Constant(0.))
  Cons2.append(Constant(0.))
  Fform += forceall([Cons0[i],Cons1[i],Cons2[i]],test,dpact(i+1))
#############################





for step in range(int(np.shape(steps)[0]/2.3)):
#for step in range(200):
  print step
  ff.u = steps[step,:]
  if step == 0: 
    ff.uold = np.zeros(np.size(ff.u))
  else:
    ff.uold = steps[step-1,:]
    
    ##################OXY###########################
    oxy = oxystoring.sum(1)/oxycount
    foxy = foxystoring.sum(1)/oxycount 
    ff.uold[:] = oxy[:]
    print 'oxy'
    oxystoring = np.zeros(n_act).reshape(n_act,1)
    foxystoring = np.zeros(n_act).reshape(n_act,1)
    oxycount = 0.
    oxy = np.zeros(n_act)
    #################################################

  
  #1st step (eulero)^2
  time = dt
  itime = idt
  localtime = np.array([])
  if step == 0:
    eu.initialize(np.zeros(np.shape(F.array())),np.zeros(np.shape(F.array())))
    amp = act.force2forces_all(ff.force(time)+fcostvector, eu.u, offx, offy)
    for i in range(n_act): 
      delta = PointSource(L.sub(0), act.point(i), amp[i,0])
      delta.apply(F)
      delta = PointSource(L.sub(1), act.point(i), amp[i,1])
      delta.apply(F)
      delta = PointSource(L.sub(2), act.point(i), amp[i,2])
      delta.apply(F)
    eu.integrate(F)
    p = plot(eu.u.sub(0),title="shape", range_max = 7.5e-6 , range_min = -7.5e-6, window_width = 1920 , window_height = 1080)
    itime +=idt
    mnt.initialize(eu.u.vector(), eu.ud.vector(), eu.udd.vector(),\
                 eu.uold.vector(),eu.udold.vector(),eu.uddold.vector())
    utemp = act.disp2disp_all(eu.u)
    udtemp = np.zeros(n_act)
    amp = ff.force(time)+fcostvector
  while itime <= T: 
    #leggo e integro costantemente la lettura
    cap = act.disp2cap_all(mnt.u)
    sens.integrate(cap)
    cap = sens.x
    u = act.cap2disp_all(cap)
    ##integro e applico costantemente la forza (per comodita' solo quella di feedback)
    curr[:]  = act.force2current_all(amp)[:]
    voltemp[:] = (Res+Rp)*curr
    
    ####TENSIONE####################################################################################
    #Current.integrate(voltemp-Rp*itemp-Ki*mnt.ud.sub(0).compute_vertex_values()[act.vertex_id])
    #curr[:] = Current.x
    ################################################################################################
    
    ireq = np.append(ireq,curr[0])
    filt.integrate(curr)
    curr[:] = filt.x[:]
    itrace = np.append(itrace,curr[0])
    itv = np.append(itv,itime+step*T)
    f_filt = act.current2force_all(curr, mnt.u, offx , offy)
    Fold[:] = F[:]
    for i in range(n_act):	
        Cons0[i].assign(f_filt[i,0])
	Cons1[i].assign(f_filt[i,1])
	Cons2[i].assign(f_filt[i,2])
    F = assemble(Fform)
    F[:] += fvect
    if itime >= time-itime/10:
      ########################OXY###############################################################
      if time>3*T/4:
	oxycount += 1
        oxystoring = np.transpose(np.append(np.transpose(oxystoring), np.transpose(utemp)).\
	      reshape(oxycount+1,n_act))
	foxystoring = np.transpose(np.append(np.transpose(foxystoring), np.transpose(amp)).\
	      reshape(oxycount+1,n_act))
      cforce = fb.force(utemp,udtemp,ff.shapew(time),ff.shapewd(time))
      if step<5:
        oforce = ff.force(time)
        amp = cforce+oforce+fcostvector
      else:
        oforce = ff.oxyforce(time,foxy)
        amp = cforce+oforce
      
      ##########################################################################################
      ########################NON OXY###########################################################
      #oforce = ff.force(time)
      #cforce = fb.force(utemp,udtemp,ff.shapew(time),ff.shapewd(time))
      #amp = cforce+oforce+fcostvector
      ##########################################################################################
      
      #itemp[:] = digital_current(Current.x)
      itemp[:] = Current.x




      #####differenze finite
      #udtemp[:] = (u[:]-utemp[:])/dt
      #####filtro la posizione e faccio le differenze finite
      my_filter_for_vel.integrate(u[:])
      udtemp[:] = (my_filter_for_vel.x[:]-my_filter_for_vel.xold[:])/dt
      #####filtro 2o ordine
      ##print u
      ##print w_k
      #w_k1[:] = w_k[:]
      #v_k[:] += L2*(u[:]-w_k[:])
      #w_k[:] += +L1*(u[:]-w_k[:])+dt*v_k[:]
      ##udtemp[:] = v_k
      #udtemp[:] = (w_k[:]-w_k1[:])/dt
      ##print udtemp
      
      
      #M11=0.9675
      #M21=-10.7228
     
      M11=0.7015
      M21=-6.8291

      #M11=0.398010674536167
      #M21=-1.93969927673054

      A11 = 0.9209
      A12 = -0.06597
      B11 = 0.0326
      
      iobserver_old =iobserver  + M11*(itemp-iobserver)
      vobserver_old = vobserver + M21*(itemp-iobserver)
      
      iobserver = A11*iobserver_old+A12*vobserver_old+B11*(voltemp-Rp*itemp)
      vobserver = vobserver_old
      
      
      
      utemp[:] = u[:]
    mnt.integrate(F)
    mnt.update()
    p.plot(mnt.u.sub(0))


    if itime >= time-itime/10:  
      localtime = np.append(localtime,time)
      trace = np.transpose(np.append(np.transpose(trace), np.transpose(act.disp2disp_all(mnt.u))).\
	      reshape(np.shape(trace)[1]+1,np.shape(trace)[0]))
      tracetrue = np.transpose(np.append(np.transpose(tracetrue), np.transpose(mnt.u.sub(0).compute_vertex_values()[act.vertex_id])).\
	      reshape(np.shape(tracetrue)[1]+1,np.shape(tracetrue)[0]))
      tracevel = np.transpose(np.append(np.transpose(tracevel), np.transpose(udtemp)).\
	      reshape(np.shape(tracevel)[1]+1,np.shape(tracevel)[0]))
      traceveltrue = np.transpose(np.append(np.transpose(traceveltrue), np.transpose(mnt.ud.sub(0).compute_vertex_values()[act.vertex_id])).\
	      reshape(np.shape(traceveltrue)[1]+1,np.shape(traceveltrue)[0]))
      tv = np.append(tv,itime+step*T)
      
      iobserver_vect = np.transpose(np.append(np.transpose(iobserver_vect), np.transpose(iobserver_old)).\
	      reshape(np.shape(iobserver_vect)[1]+1,np.shape(iobserver_vect)[0]))
      vobserver_vect = np.transpose(np.append(np.transpose(vobserver_vect), np.transpose(vobserver_old)).\
	      reshape(np.shape(vobserver_vect)[1]+1,np.shape(vobserver_vect)[0]))
      tracevoltage = np.transpose(np.append(np.transpose(tracevoltage), np.transpose(voltemp)).\
	      reshape(np.shape(tracevoltage)[1]+1,np.shape(tracevoltage)[0]))      
      tracecurrent = np.transpose(np.append(np.transpose(tracecurrent), np.transpose(curr)).\
	      reshape(np.shape(tracecurrent)[1]+1,np.shape(tracecurrent)[0]))
      time += dt

      with open('results/utemp.dat', 'a') as fid:
         fid.write(" ".join(map(str, act.disp2disp_all(mnt.u))))
         fid.write("\n")
      with open('results/utruetemp.dat', 'a') as fid:
         fid.write(" ".join(map(str, mnt.u.sub(0).compute_vertex_values()[act.vertex_id])))
         fid.write("\n")
      with open('results/udtemp.dat', 'a') as fid:
         fid.write(" ".join(map(str, udtemp)))
         fid.write("\n")
      with open('results/udtruetemp.dat', 'a') as fid:
         fid.write(" ".join(map(str, mnt.ud.sub(0).compute_vertex_values()[act.vertex_id])))
         fid.write("\n")
      with open('results/timetemp.dat', 'a') as fid:
         fid.write("%.8f \n" % (itime+step*T))
    itime += idt
    
  ref = np.transpose(np.append(np.transpose(ref), np.transpose(ff.get_all_stories(localtime))).\
	      reshape(np.shape(ref)[1]+np.size(localtime),np.shape(ref)[0]))
  refvel = np.transpose(np.append(np.transpose(refvel), np.transpose(ff.get_all_stories_vel(localtime))).\
	      reshape(np.shape(refvel)[1]+np.size(localtime),np.shape(refvel)[0]))    


np.savetxt('cu.dat', np.append(tv*1000,trace[0]*1e9).reshape(2,np.size(tv)).transpose())
np.savetxt('cutrue.dat', np.append(tv*1000,tracetrue[0]*1e9).reshape(2,np.size(tv)).transpose())
np.savetxt('cud.dat', np.append(tv*1000,tracevel[0]).reshape(2,np.size(tv)).transpose())
np.savetxt('cudtrue.dat', np.append(tv*1000,traceveltrue[0]).reshape(2,np.size(tv)).transpose())
#np.savetxt('ciobserved.dat', np.append(tv*1000,iobserver_vect[0]).reshape(2,np.size(tv)).transpose())
#np.savetxt('cudobserved.dat', np.append(tv*1000,vobserver_vect[0]).reshape(2,np.size(tv)).transpose())
#np.savetxt('citrue.dat', np.append(tv*1000,tracecurrent[0]).reshape(2,np.size(tv)).transpose())
#np.savetxt('cvtrue.dat', np.append(tv*1000,tracevoltage[0]).reshape(2,np.size(tv)).transpose())

#embed()
#np.savetxt("results/ref3.dat",ref[0,:])
#np.savetxt("results/refd3.dat",refvel[0,:])
#np.savetxt("results/u3.dat",trace[0,:])
#np.savetxt("results/utrue3.dat",tracetrue[0,:])
#np.savetxt("results/ud3.dat",tracevel[0,:])
#np.savetxt("results/udtrue3.dat",traceveltrue[0,:])
#np.savetxt("results/time3.dat",tv)
#np.savetxt("results/orizzontal3.dat",np.transpose(np.append(offx,offy).reshape(2,n_act)))
fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(tv, trace[0,:],  c='r', marker = '8')
ax.plot(tv, tracetrue[0,:],  c='g', marker = '8')
ax.plot(tv, ref[0,:]+1e-8,  c='b')
ax.plot(tv, ref[0,:]-1e-8,  c='b')
ax.plot(tv, ref[0,:]+2e-8,  c='b')
ax.plot(tv, ref[0,:]-2e-8,  c='b')
ax.plot(tv, ref[0,:], c = 'b')
ax2 = fig.add_subplot(312)
ax2.plot(tv, tracevel[0,:],  c='r', marker = '8')  
ax2.plot(tv, traceveltrue[0,:],  c='g', marker = '8')  
ax2.plot(tv, vobserver_vect[0,:],  c='b', marker = '8')  
ax2.plot(tv, refvel[0,:]+1e-8,  c='b', ) 
ax2.plot(tv, refvel[0,:]-1e-8,  c='b') 
ax2.plot(tv, refvel[0,:]+2e-8,  c='b')
ax2.plot(tv, refvel[0,:]-2e-8,  c='b')
ax2.plot(tv, refvel[0,:], c = 'b')
ax3 = fig.add_subplot(313)
ax3.plot(tv, tracecurrent[0,:],  c='r', marker = '8')
ax3.plot(tv, iobserver_vect[0,:],  c='b', marker = '8')
plt.show()
