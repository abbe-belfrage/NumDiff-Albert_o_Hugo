import numpy as np
import scipy as sp
import scipy . linalg
import matplotlib . pyplot as plt

class ivp: 
    def __init__ (self, t0, T, y0):
        self.t0 = t0
        self.T = T
        self.y0 = y0
        return self

    def f (self, t, y):
        raise NotImplementedError

class LTE:
    def __init__ (self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f (self, t, y ) :
        return self.A @ y

    def exact(self, t):
        return scipy.linalg.expm(t*self.A)

ivp = LTE(t0 = 0 , T = 1, y0 = np.array([2]) , A = np.array([[-1]]))
#solver = ExplicitEuler(ivp, N = 100)
#solver.integrate()
# Plot the approximations { u_n } vs . the temporal
# grid points { t_n }:
#plt . plot ( solver . grid , solver . us )

ts = np.linspace(0, 1)
ys = [ivp.exact(t) for t in ts ]
plt.plot(ts, ys)