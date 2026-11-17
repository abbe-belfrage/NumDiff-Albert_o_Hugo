import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
# import sys
# sys.path.append("..")
from Classes import Solver, ivp, LTE, Evaluator, LotkaVolterra, vanderPol

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
        self.step(self.grid[-1], self.u)
        hn = ((self.tol / self.errest) ** (2 / (3*self.k))) * ((self.tol / self.errestold) ** (-1 / (3*self.k)) * self.h)
        self.h = hn                                    

    def integrate(self):
        if self.adaptive == True:
            T = self.ivp.T
            t0 = self.ivp.t0
            us = np.array([self.u])
            self.grid = np.append(self.grid, t0)
            self.h = np.abs(T - t0) * self.tol ** (1 / self.k) / (100 * (1 + np.linalg.norm(self.ivp.f(t0, self.ivp.y0))))
            while self.grid[-1] < T:
                t_curr = self.grid[-1]
                self.newstep()
                if t_curr + self.h > T:
                    self.h = T - self.grid[-1]
                    t_next = T
                else: 
                    t_next = t_curr + self.h
                self.u = self.step(t_curr, self.u)
                self.grid = np.append(self.grid, t_next)
                us = np.append(us, self.u)
        else: 
            us = np.zeros(self.N + 1)
            for i in range(self.N):
                us[i] = self.u
                self.u = self.step(self.grid[i], self.u)
                us[self.N] = self.u
        return self.grid, us

mu = 100
ivp = vanderPol(t0 = 0, T = 2*mu, y0 = np.array([2, 0]), mu = mu)
solver = RK34(ivp, 100, tol = 1e-6, adaptive=True)
ts, ys = solver.integrate()
y_2s = ys[1::2]
y_1s = ys[0::2]
plt.plot(ts, y_2s, label = 'y_2 over time')
plt.xlabel('time')
plt.ylabel('y_2')
plt.legend()
plt.plot(y_1s, y_2s, label = 'Phase portrait')
plt.xlabel('y_1')
plt.ylabel('y_2')
plt.legend()
plt.show()

# ivp = LotkaVolterra(t0 = 0, T = 100, y0 = np.array([1, 0.9]), a = 3, b = 9, c = 15 , d = 15)      #Task3
# solver = RK34(ivp, 100000, tol = 1e-6, adaptive = True)
# ts, us = solver.integrate()
# xs = us[0::2]  # Even indices: x0, x1, x2, ... (Rabbits)
# ys = us[1::2]  # Odd indices:  y0, y1, y2, ... (Foxes)
# # plt.plot(ts, xs, label = 'Rabbits')                     #Task3.2, 3.3
# # plt.plot(ts, ys, label = 'Foxes')
# # plt.xlabel('time')
# # plt.ylabel('# of animals')
# # plt.legend()
# # plt.show()

# # plt.plot(xs, ys)                                        
# # plt.xlabel('Rabbits (x)')
# # plt.ylabel('Foxes (y)')
# # plt.title('Lotka-Volterra Phase Portrait')
# # plt.show()


# ivp = LTE(t0 = 0, T = 1, y0 = 2, A = -1)            #For Task 2.5
# solver = RK34(ivp, 1000, tol=1e-6, adaptive=False)
# ts, us = solver.integrate()
# plt.plot(ts, us, label = 'RK34 fixed')
# solver_2 = RK34(ivp, 1000, tol=1e-6, adaptive=True)
# ts, us = solver_2.integrate()
# plt.plot(ts, us, label = 'RK34 adaptive')
# plt.xlabel('ts')
# plt.ylabel('us')
# plt.legend()
# plt.show()


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