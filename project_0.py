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

class LTE(ivp):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t0, y ):
        return self.A @ y

    def exact(self, t):
        return scipy.linalg.expm(t*self.A)

class Solver:
    def __init__(self, ivp, N):
        self.ivp = ivp
        self.N = N
        self.u = ivp.y0
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
        self.h = (ivp.T - ivp.t0) / N
        return self

    def step(self, tn, un):
        raise NotImplementedError

    def integrate(self):
        us = np.zeros(self.N + 1)
        for i in range(self.N):
            us[i] = self.u
            self.u = self.step(self.grid[i], self.u)
        us[self.N] = self.u
        return self.grid , us

class ExplicitEuler(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        return un + self.h * self.ivp.f(tn, un)

ivp = LTE(t0 = 0 , T = 1, y0 = np.array([2]) , A = np.array([[-1]]))
solver = ExplicitEuler(ivp, N = 100)
ts, us = solver.integrate()
# Plot the approximations { u_n } vs . the temporal
# grid points { t_n }:
plt.plot(ts, us)
plt.show()

ts = np.linspace(0, 1)
ys = [ivp.exact(t) @ ivp.y0 for t in ts ]
plt.plot(ts, ys)
plt.show()