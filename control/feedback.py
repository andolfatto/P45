from dolfin import *
import numpy as np
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D      
from actuators import *
import matplotlib.pyplot as plt       
from integrators import *

class FeedBack:
  def __init__(self, actlist, dt, gp = 10000*np.ones(45), gd = 35*np.ones(45),capacitor_emulate = False):
    "actlist e' elemento della classe ActList con i dati degli attuatori,gp e gd sono i guadagni del pd (vettori)"
    self.al = actlist
    self.gp = gp
    self.gd = gd
    self.capacitor_emulate = capacitor_emulate
    self.sens = SensorFilter(40000, np.size(self.al.act_id), dt, 3.188993986668676e-11)
  def force(self,uold,udold,uref,udref):
    "uold, udold sono i campi completi al passo precedente"
    "uref, udref sono i vettori di riferimento al passo precedente nei punti attuati"
    "capacitor_emulate attiva la simulazione della lettura del condensatore"
    if self.capacitor_emulate:
      u = self.al.disp2disp_all(uold)
     
      ###attivare questo blocco per avere il filtro passa-basso sulla lettura del condensatore
      #############################################
      #cap = self.al.disp2cap_all(uold)
      #self.sens.integrate(cap)
      #c = self.sens.x
      #u = self.al.cap2disp_all(c)
      #############################################
      
      
    else:
      u  =  uold.sub(0).compute_vertex_values()[self.al.vertex_id]
    ud = udold.sub(0).compute_vertex_values()[self.al.vertex_id]
    return self.gp*(uref-u)+self.gd*(udref-ud)
   