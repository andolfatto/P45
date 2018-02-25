import numpy as np
from mpl_toolkits.mplot3d import Axes3D                                          
import matplotlib.pyplot as plt  
from IPython import embed

class SensorFilter:
  def __init__(self, ft, n, dt, x0):
    "ft e' la frequenza di taglio, n il numero di sensori"
    self.dt = dt
    self.ft = ft
    self.n = n
    self.x = x0*np.ones(n)
    self.w = (2*np.pi*ft)
  def integrate(self, u):
    "u e' l'array di ingresso del sistema xd = -x/w+u/w"
    self.x = (self.x/self.dt+u*self.w)/(1/self.dt+self.w)




pippo = SensorFilter(4,1,1e-3,0)
time = np.linspace(0,1,1e3)
u = np.ones(np.size(time))
x = np.zeros(np.size(time))
for i in range(np.size(time)):
  x[i] = pippo.x[0]
  pippo.integrate(u)

#embed()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time, u,  c='r')
ax.plot(time, x,  c='b')
plt.show()