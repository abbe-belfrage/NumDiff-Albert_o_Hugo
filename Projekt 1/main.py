import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
# import sys
# sys.path.append("..")
from Classes import Solver, ivp, LTE, Evaluator

class RK4(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        h = self.h
        Y_1_prime = self.ivp.f(tn, un)
        Y_2_prime = self.ivp.f(tn + h/2, un + h * Y_1_prime / 2)
        Y_3_prime = self.ivp.f(tn + h/2, un + h * Y_2_prime / 2)
        Y_4_prime = self.ivp.f(tn + h, un + h * Y_3_prime)

        un = un + h / 6 * (Y_1_prime + 2 * Y_2_prime + 2 * Y_3_prime + Y_4_prime)
        return un

    def integrate(self):
        us = np.zeros(self.N + 1)
        for i in range(self.N):
            us[i] = self.u
            self.u = self.step(self.grid[i], self.u)
            us[self.N] = self.u
        return self.grid, us


class RK34(Solver):
    def __init__(self, ivp, N, tol, adaptive):
        super().__init__(ivp, N, tol, adaptive)
        self.errest = None
        self.errestold = None
        self.tol = tol
        self.k = 4                  # RK34 has k = 4 (according to science)'
        

    def step(self, tn, un):
        h = self.h
        Y_1_prime = self.ivp.f(tn, un)
        Y_2_prime = self.ivp.f(tn + h/2, un + h * Y_1_prime / 2)
        Y_3_prime = self.ivp.f(tn + h/2, un + h * Y_2_prime / 2)
        Z_3_prime = self.ivp.f(tn + h, un - h * Y_1_prime + 2 * h * Y_2_prime)
        Y_4_prime = self.ivp.f(tn + h, un + h * Y_3_prime)

        un = un + h / 6 * (Y_1_prime + 2 * Y_2_prime + 2 * Y_3_prime + Y_4_prime)
        self.errest = np.linalg.norm(h / 6 * (2 * Y_2_prime + Z_3_prime - 2 * Y_3_prime - Y_4_prime))
        return un

    def newstep(self):
        if self.errestold == None:
            self.errestold = self.tol
        else:
            self.errestold = self.errest
        self.step(self.grid, self.u)
        hn = ((self.tol / self.errest) ** (2 / (3*self.k))) * ((self.tol / self.errestold) ** (-1 / (3*self.k)) * self.h)
        self.h = hn                                    

    def integrate(self):
        if self.adaptive == True:
            T = self.ivp.T
            t0 = self.ivp.t0
            us = np.zeros(0)
            self.grid = np.append(self.grid, t0)
            us = np.append(us, self.u)
            self.h = np.abs(T - t0) * self.tol ** (1 / self.k) / (100 * (1 + np.linalg.norm(self.ivp.f(t0, self.ivp.y0))))
            while self.grid[-1] < self.ivp.T:
                if self.grid[-1] > T:
                    self.h = T - self.grid[-1]
                self.u = self.step(self.grid[-1] + self.h, self.u)
                us = np.append(us, self.u)
                self.grid = np.append(self.grid, self.grid[-1] + self.h)
                self.newstep()
        else: 
            us = np.zeros(self.N + 1)
            for i in range(self.N):
                us[i] = self.u
                self.u = self.step(self.grid[i], self.u)
                us[self.N] = self.u
        return self.grid, us


ivp = LTE(t0 = 0, T = 1, y0 = 2, A = -1)            #For Task 2.2
solver = RK34(ivp, 1000, tol=1e-6, adaptive=True)
ts, us = solver.integrate()
#exact = solver.ivp.exact(ts)
plt.plot(ts, us, label = 'RK34 adaptive')
#plt.plot(ts, exact, label = 'exact')
plt.xlabel('ts')
plt.ylabel('us')
plt.legend()
plt.show()
print(solver.grid)


#ivp = LTE(t0 = 0, T = 1, y0 = 2, A = -1)            #For Task 2.2
#evaluator = Evaluator(ivp, RK4)
#Ns = [10, 100, 1000]
#errs, grids = evaluator.errvsh(Ns, True)
#log_errs = np.log10(errs)
#log_grids = np.log10(grids)
#plt.plot(log_grids, log_errs, label = 'RK4 loglog error')
#plt.xlabel('log_ts')
#plt.ylabel('log_grids')
#plt.legend()
#plt.show()