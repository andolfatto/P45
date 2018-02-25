from dolfin import *
import numpy as np
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D                                          
import matplotlib.pyplot as plt                                                  


    
class Pinpoint(SubDomain):
    def __init__(self, coords):
        self.coords = np.array(coords)
        SubDomain.__init__(self)
    def move(self, coords):
        self.coords[:] = np.array(coords)
    def inside(self, x, on_boundary):
        return np.linalg.norm(x-self.coords) < 1e-5


def disp2cap(w, tx, ty):
    "Computes capacitance (nF) using nodal displacement and rotations"
    cap_z = 310.e-9
    cap_a1 = 9.008809214779535e-11
    c0 = 3.188993986668676e-11
    return cap_z*w + cap_a1*(tx**2+ty**2) + c0
  
def cap2disp(cap):
    "Emulates sensor capacitance (nF) reading; gives displacement dz (m)"
    cap_z = 310.e-9
    c0 = 3.188993986668676e-11
    return (cap-c0)/cap_z
 
 
def digital_cap2disp(C):
  "Emulates sensor capacitance (nF) reading; gives displacement dz (m) including AD conv and noise"
  Cstray = 3.9e-12
  Vref = 1
  Cref = 39.e-12
  v_an = Vref/Cref*(C+Cstray)
  nbit = 15
  v_dig = np.round(2**(nbit) / Vref * v_an)
  noise_var = 1.1+2.78e-24*pow(v_dig,5)
  noise_dig = np.round(np.random.normal(0.,noise_var))
  v_dig += noise_dig
  v_an = v_dig * Vref / 2**(nbit)
  cap =  v_an / Vref *Cref - Cstray
  return cap2disp(cap)


class ActList:
  def __init__(self, V, coords):
    ActList.space = V
    if not self.space.num_sub_spaces() == 3:
      dolfin_error('capacitanceread.py', 'build actuator stuff', 'V must have 3 subspaces (w, thx, thy)')
    ActList.coords = np.array(coords)
    if not np.shape(self.coords)[1] == 2:
      dolfin_error('capacitanceread.py', 'build actuator stuff', 'coord matrix must be of shape (n,2)')
    ActList.mesh_dofx = self.space.sub(0).mesh().coordinates()[:,0]
    ActList.mesh_dofy = self.space.sub(0).mesh().coordinates()[:,1]
    ActList.dofx = self.coords[:,0]
    ActList.dofy = self.coords[:,1]
    ActList.act_num = np.size(self.dofx)
    ActList.act_id = np.array([])
    ActList.vertex_id = np.array([])
    for i in range(np.size(self.dofx)):
      for j in range(np.size(self.mesh_dofx)):
        if self.mesh_dofx[j]==self.dofx[i] and self.mesh_dofy[j]==self.dofy[i]:
          self.act_id = np.append(self.act_id,i)
          self.vertex_id = np.append(self.vertex_id,j)
    if not np.size(self.vertex_id) == self.act_num:
      dolfin_error('capacitanceread.py', 'build actuator stuff', 'given coordinates do not match with mesh')
    self.vertex_id = self.vertex_id.astype(int)
    self.act_id = self.act_id.astype(int)
    
  def plot_coords(self):
    '...'
    fig = plt.figure()                                                               
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(self.dofx, self.dofy, np.zeros(np.size(self.dofx)), c='b', marker='.')     
    plt.show()  
    
  def plot_displacement(self, sol):
    "..."     
    if not np.size(sol.sub(0).compute_vertex_values()) == np.size(self.mesh_dofx):
      dolfin_error('capacitanceread.py', 'plot actuator displacements', 'wrong vertex number in given solution')
    u = sol.sub(0).compute_vertex_values()[self.vertex_id]
    fig = plt.figure()                                                               
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(self.dofx, self.dofy, u, c='b', marker='.') 
    plt.show()  
    
  def disp2cap_all(self, sol):
    "..."
    if not np.size(sol.sub(0).compute_vertex_values()) == np.size(self.mesh_dofx):
      dolfin_error('capacitanceread.py', 'computing capacitance', 'wrong vertex number in given solution')
    w = sol.sub(0).compute_vertex_values()[self.vertex_id]
    tx =sol.sub(1).compute_vertex_values()[self.vertex_id]
    ty =sol.sub(2).compute_vertex_values()[self.vertex_id] 
    cap_all = np.array([])
    for i in range(np.size(w)):
      cap_all = np.append(cap_all, disp2cap(w[i],tx[i],ty[i]))
    return cap_all
  
  def cap2disp_all(self, cap_vect):
    "..."
    w = np.array([])
    for i in range(np.size(cap_vect)):
      w = np.append(w,digital_cap2disp(cap_vect[i]))
    return w
  
  def cap2disp_all_clean(self, cap_vect):
    "..."
    w = np.array([])
    for i in range(np.size(cap_vect)):
      w = np.append(w,cap2disp(cap_vect[i]))
    return w
  
  def disp2disp_all(self, sol):
    "..."
    return self.cap2disp_all(self.disp2cap_all(sol))
  
  def disp2disp_all_clean(self, sol):
    "..."
    return self.cap2disp_all_clean(self.disp2cap_all(sol))
  def pinpoint(self,i):
    "..."
    return Pinpoint(self.coords[i,:])
  
  def mark(self,subdomains,val,i,):
    "..."
    p = self.pinpoint(i)
    p.mark(subdomains,val)
  
  def mark_all(self,subdomains,val):
    "..."
    for i in self.act_id:
     p = self.pinpoint(i)
     p.mark(subdomains,val)
  
  def dp(self):
    "i e' l'indice dell'attuatore. Per richiamare il dp l'indice e' i+1 (1 per il primo attuatore,ecc)"
    lumped_boundaries = MeshFunction("size_t", self.space.mesh(), 0)
    lumped_boundaries.set_all(0)
    for i in self.act_id:
      self.mark(lumped_boundaries,i+1,i)
    #plot(lumped_boundaries)
    #interactive()
    return Measure('vertex')[lumped_boundaries]
  
  def dp_all(self):
    "punta a tutti gli attuatori insieme. Indice 1"
    lumped_boundaries = MeshFunction("size_t", self.space.mesh(), 0)
    lumped_boundaries.set_all(0)
    for i in self.act_id:
      self.mark(lumped_boundaries,1,i)
    return Measure('vertex')[lumped_boundaries]
  
  def point(self,i):
    return Point(self.coords[i,0],self.coords[i,1])
