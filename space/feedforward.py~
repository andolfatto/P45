from dolfin import *
import numpy as np
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D                                          
import matplotlib.pyplot as plt                                                  

class FeedForward:
  def __init__(self,u,uold,T, actlist, capacitanceread = False):
    "u e' il vettore degli spostamenti desiderati, T e' il periodo per il quale il comando viene mantenuto"
    "capacitance_read uses the distorted reading of capacitance"
    self.u = u
    self.uold = uold
    self.T = T
    self.al = actlist
    if capacitanceread:
      self.K = np.load("data/Kred_with_cond.npy")
    else:
      self.K = np.load("data/Kred.npy")
    self.M = np.load("data/Mred.npy")
    self.alfa = 1e-3
    self.beta = 1e-3
    self.C = self.alfa*self.M+self.beta*self.K
    
  def recompute_C(self):
    "..."
    self.C = self.alfa*self.M+self.beta*self.K
    
  def shapew(self,t):
    if t < self.T/4:
      y = .5*(1-np.cos(4*np.pi/self.T * t))
    else:
      y = 1
    return self.uold + y*(self.u-self.uold)

  def shapewd(self,t):
    "..."
    if t < self.T/4:
      yd = 2*np.pi/self.T * np.sin(4*np.pi/self.T *t)
    else:
      yd = 0
    return yd*(self.u-self.uold)
  
  def shapewdd(self,t):
    "..."
    if t < self.T/4:
      ydd= 8*(np.pi/self.T)**2 * np.cos(4*np.pi/self.T *t)
    else: 
      ydd = 0
    return ydd*(self.u-self.uold)
  
  def force(self,t):
    "..."
    return (self.M.dot(self.shapewdd(t))+self.C.dot(self.shapewd(t))+self.K.dot(self.shapew(t)))

  def oxyforce(self,t,ff):
    "..."
    if t < self.T/4:
      y = .5*(1-np.cos(4*np.pi/self.T * t))
    else:
      y = 1
    return (self.M.dot(self.shapewdd(t))+self.C.dot(self.shapewd(t))+self.K.dot(y*(self.u-self.uold))+ff)

  def plot_story(self,i):
    'plots shape of i-th point displacement'
    time = np.linspace(0,self.T,10000)
    sh = np.zeros(np.shape(time))
    shd = np.zeros(np.shape(time))
    shdd = np.zeros(np.shape(time))
    for j in range(np.size(time)):
      sh[j] = self.shapew(time[j])[i]
      shd[j] = self.shapewd(time[j])[i]
      shdd[j] = self.shapewdd(time[j])[i]
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(time, sh, c='r')                                                                      
    ax2 = fig.add_subplot(312)
    ax2.plot(time, shd, c='g')                                                         
    ax3 = fig.add_subplot(313)
    ax3.plot(time, shdd, c='b')                  
    plt.show()
    
  def get_story(self,i,time):
    'plots shape of i-th point displacement'
    sh = np.zeros(np.shape(time))
    for j in range(np.size(time)):
      sh[j] = self.shapew(time[j])[i]
    return sh
  
  def get_all_stories(self,time):
    '...'
    v = np.array([])
    for i in np.arange(np.size(self.u)):
      v = np.append(v, self.get_story(i,time))
    return v.reshape(np.size(self.u),np.size(time))

  
  def get_story_vel(self,i,time):
    'plots shape of i-th point displacement'
    sh = np.zeros(np.shape(time))
    for j in range(np.size(time)):
      sh[j] = self.shapewd(time[j])[i]
    return sh
  
  def get_all_stories_vel(self,time):
    '...'
    v = np.array([])
    for i in np.arange(np.size(self.u)):
      v = np.append(v, self.get_story_vel(i,time))
    return v.reshape(np.size(self.u),np.size(time))


  def weird_force(self, t, sol, dx = np.zeros(np.size(45)) , dy = np.zeros(np.size(45)) ):
    "current control modification on force calculation"
    return self.al.force2forces_all(self.force(t),sol, dx, dy)
