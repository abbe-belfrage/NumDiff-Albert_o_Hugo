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
        self.errest = h / 6 * (2 * Y_2_prime + Z_3_prime - 2 * Y_3_prime - Y_4_prime)
        return un

    def newstep(self):
        if self.errestold == None:
            self.errestold = self.tol
        else:
            self.errestold = self.errest
        self.step()
        hn = ((self.tol / self.errest) ** (2 / (3*self.k))) * ((self.tol / self.errestold) ** (-1 / (3*self.k)) * self.h)
        
        self.h = hn                                    

    def integrate(self):
        if self.adaptive == True:
            us = np.zeros(self.N + 1)
            sum_h = 0
            T = self.ivp.T
            t0 = self.ivp.t0
            self.h = np.abs(T - t0) * self.tol ** (1 / self.k) / (100 * (1 + np.linalg.norm(self.ivp.f(t0, self.ivp.y0))))
            if sum_h < self.ivp.T:

            else: 

        else: 
            us = np.zeros(self.N + 1)
            for i in range(self.N):
                us[i] = self.u
                self.u = self.step(self.grid[i], self.u)
                us[self.N] = self.u
            return self.grid, us


ivp = LTE(t0 = 0, T = 1, y0 = 2, A = -1)            #For Task 2.2
evaluator = Evaluator(ivp , RK4)
Ns = [10, 100, 1000]
errs, grids = evaluator.errvsh(Ns, False)
log_errs = np.log10(errs)
log_grids = np.log10(grids)
plt.plot(log_grids, log_errs, label = 'RK4 loglog error')
plt.xlabel('log_ts')
plt.ylabel('log_grids')
plt.legend()
plt.show()

Testsolver = Solver.__init__
print(Testsolver)