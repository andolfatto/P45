from dolfin import *
import numpy as np
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D      
from actuators import *
import matplotlib.pyplot as plt                                                  

class Eulero:
  
  def __init__(self, V, M, C, K, dt):
    "V is the function space, M,C,K are the assembled matrices, dt is the tipe step (can be changed)"
    self.V = V
    self.dt = dt
    self.M = M
    self.C = C
    self.K = K
      
  def initialize(self, u0, v0):
    self.LU = LUSolver(self.M/(self.dt**2)+self.C/self.dt+self.K)
    self.LU.parameters["reuse_factorization"] = True
    self.LU.parameters["verbose"] = True
    self.u = Function(self.V)
    self.ud = Function(self.V)
    self.udd = Function(self.V)
    self.uold = Function(self.V)
    self.udold = Function(self.V)
    self.uddold = Function(self.V)
    self.uold.vector()[:] = u0
    self.udold.vector()[:] = v0
  def integrate(self, F):
    "..."
    self.LU.solve(self.u.vector(), \
           F +(self.M/(self.dt**2)+self.C/self.dt)*self.uold.vector()+\
	       self.M/self.dt*self.udold.vector())
    self.ud.assign((self.u-self.uold)/self.dt )
    self.udd.assign((self.ud-self.udold)/self.dt )
      
  def update(self):
    "update old values"
    self.uold.assign(self.u)
    self.udold.assign(self.ud)
    self.uddold.assign(self.udd)
    


   
   
class Mantegazza:
  
  def __init__(self, V, M, C, K, dt, rho = .58, alpha = 1.):
    "V is the function space, M,C,K are the assembled matrices, dt is the tipe step (can be changed)"
    self.V = V
    self.dt = dt
    self.M = M
    self.C = C
    self.K = K
    self.rho = rho
    self.alpha = alpha
    self.D = 2.*(1.+self.alpha)-(1.-self.rho)**2
    self.beta = self.alpha*((1.-self.rho)**2*(2.+self.alpha)+\
                2*(2.*self.rho -1.)*(1.+self.alpha))/self.D
    self.delta = 0.5*self.alpha**2*(1.-self.rho)**2/self.D
    self.a0 = 1-self.beta
    self.a_1 = self.beta
    self.b1 = self.dt*(self.delta/self.alpha+self.alpha/2)
    self.b0 = self.dt*(self.beta/2+self.alpha/2-(1+self.alpha)*self.delta/self.alpha)
    self.b_1 = self.dt*(self.beta/2+self.delta)

  def initialize(self, u_1, ud_1, udd_1, u_2, ud_2, udd_2):
    "voglio come input i vettori iniziali (rispettivamente al passo k-1 e k-2)"
    left = self.M + self.K*self.b1*self.b1 +self.C*self.b1
    self.LU = LUSolver(left)
    self.LU.parameters["reuse_factorization"] = True
    self.LU.parameters["verbose"] = True
    self.u = Function(self.V)
    self.ud = Function(self.V)
    self.udd = Function(self.V)
    self.u_1 = Function(self.V)
    self.ud_1 = Function(self.V)
    self.udd_1 = Function(self.V)
    self.u_2 = Function(self.V)
    self.ud_2 = Function(self.V)
    self.udd_2 = Function(self.V)
    self.u_1.vector()[:] = u_1
    self.ud_1.vector()[:] = ud_1
    self.udd_1.vector()[:] = udd_1
    self.u_2.vector()[:] = u_2
    self.ud_2.vector()[:] = ud_2
    self.udd_2.vector()[:] = udd_2
  
  def integrate(self, F):
    self.u.vector()[:] = 0
    self.ud.vector()[:] = 0
    self.udd.vector()[:] = 0
    "..."
    v_kk = self.a0*self.ud_1.vector() +\
           self.a_1*self.ud_2.vector() +\
           self.b0*self.udd_1.vector() +\
           self.b_1*self.udd_2.vector()
    u_kk = self.a0*self.u_1.vector() +\
           self.a_1*self.u_2.vector() +\
           (self.b1*self.a0 + self.b0)*self.ud_1.vector() +\
           (self.b1*self.a_1 + self.b_1)*self.ud_2.vector() +\
           self.b1*self.b0*self.udd_1.vector() +\
           self.b1*self.b_1*self.udd_2.vector()
    right = F-self.K*u_kk-self.C*v_kk
    self.LU.solve(self.udd.vector(),right )
    self.ud.assign(self.a0*self.ud_1 + self.a_1*self.ud_2 +\
                   self.b1*self.udd + self.b0*self.udd_1 + self.b_1*self.udd_2)
    self.u.assign(self.a0*self.u_1 + self.a_1*self.u_2 +\
                  self.b1*self.ud + self.b0*self.ud_1 + self.b_1*self.ud_2)
      
  def update(self):
    "update old values"
    self.u_2.assign(self.u_1)
    self.u_1.assign(self.u)
    self.ud_2.assign(self.ud_1)
    self.ud_1.assign(self.ud)
    self.udd_2.assign(self.udd_1)
    self.udd_1.assign(self.udd)






class SensorFilter:
  def __init__(self, ft, n, dt, x0):
    "ft e' la frequenza di taglio, n il numero di sensori"
    self.dt = dt
    self.ft = ft
    self.n = n
    self.x = x0*np.ones(n)
    self.xold = x0*np.ones(n)
    self.w = (2*np.pi*ft)
  def integrate(self, u):
    "u e' l'array di ingresso del sistema xd = -x*w+u*w"
    #self.x = (self.x/self.dt+u*self.w)/(1/self.dt+self.w)
    self.xold[:] = self.x[:]
    self.x = (self.x+u*self.dt*self.w)/(self.w*self.dt+1)

    
    
class CurrentFilter:
  def __init__(self, R, L, n, dt):
    "ft e' la frequenza di taglio, n il numero di sensori"
    self.dt = dt
    self.R = R
    self.L = L
    self.n = n
    self.x = np.zeros(n)
    self.xold = np.zeros(n)
  def integrate(self, u):
    "u e' l'array di ingresso del sistema xd = -x*w+u*w"
    self.xold[:] = self.x[:]
    self.x = self.xold*(1-self.dt*self.R/self.L)+self.dt/self.L*u



class AdamBashforth:
  
  def __init__(self, V, mdiag, C, K, dt):
    "V is the function space, mdiag is a Function,C,K are the assembled matrices, dt is the tipe step (can be changed)"
    self.V = V
    self.dt = dt
    self.mdiag = mdiag
    self.C = C
    self.K = K
    
  def initialize(self, u_1, ud_1, u_2, ud_2):
    "voglio come input i vettori iniziali (rispettivamente al passo k-1 e k-2)"
    self.u = Function(self.V)
    self.ud = Function(self.V)
    self.u_1 = Function(self.V)
    self.ud_1 = Function(self.V)
    self.u_2 = Function(self.V)
    self.ud_2 = Function(self.V)
    self.u_1.vector()[:] = u_1
    self.ud_1.vector()[:] = ud_1
    self.u_2.vector()[:] = u_2
    self.ud_2.vector()[:] = ud_2
    self.KsuM = (self.K.array().transpose()*self.mdiag.vector()).transpose()
    self.CsuM = (self.C.array().transpose()*self.mdiag.vector()).transpose()
  
  def integrate(self, f_1, f_2):
    self.u.vector()[:] = 0
    self.ud.vector()[:] = 0
    "..."
    self.ud.vector()[:] = self.ud_1.vector()+self.dt/2*((3*f_1-f_2)-\
                                                self.KsuM.dot(3*self.u_1.vector()-self.u_2.vector())-\
                                                self.CsuM.dot(3*self.ud_1.vector()-self.ud_2.vector())\
                                                )
    self.u.assign(self.u_1 + self.dt/2*(3*self.ud_1-self.ud_2))
      
  def update(self):
    "update old values"
    self.u_2.assign(self.u_1)
    self.u_1.assign(self.u)
    self.ud_2.assign(self.ud_1)
    self.ud_1.assign(self.ud)
  
