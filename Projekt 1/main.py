import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from Classes import Solver

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


ivp = LTE(t0 = 0, T = 1, y0 = 2, A = -1)
evaluator = Evaluator(ivp , RK4)
ts, errs = evaluator.errvstime(N = 100, bool = True)
log_ts = np.log10(ts)
log_errs = np.log10(errs)
plt.plot(ts, errs)
plt.show()
