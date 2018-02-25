from dolfin import *
import numpy as np
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D      
from actuators import *
import matplotlib.pyplot as plt       
from integrators import *

class FeedBack:
  def __init__(self, actlist, dt, gp = 10000*np.ones(45), gd = 35*np.ones(45)):
    "actlist e' elemento della classe ActList con i dati degli attuatori,gp e gd sono i guadagni del pd (vettori)"
    self.al = actlist
    self.gp = gp
    self.gd = gd
    self.capacitor_emulate = capacitor_emulate
    self.dt = dt
    self.act = SensorFilter(25000, np.size(self.al.act_id), dt, 0.)
  def force(self,u,ud,uref,udref):
    "uref, udref sono i vettori di riferimento al passo precedente nei punti attuati"
    "capacitor_emulate attiva la simulazione della lettura del condensatore"
    return self.gp*(uref-u)+self.gd*(udref-ud)
   
  def weird_force(self,u,ud,uref,udref,ureal, dx = np.zeros(np.size(45)) , dy = np.zeros(np.size(45))):
    "current control modification on force calculation"
    F =  self.al.force2forces_all(self.force(u,ud,uref,udref),ureal, dx , dy)
    return F
