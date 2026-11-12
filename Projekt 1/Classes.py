import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

class ivp: 
    def __init__ (self, t0, T, y0):
        self.t0 = t0
        self.T = T
        self.y0 = y0
        return self

    def f(self, t, y):
        raise NotImplementedError

class LTE(ivp):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t0, y):            #Handles matrices correctly (hopefully, maybe)
        if np.ndim(self.A) > 1:
            return self.A @ y
        else: 
            return self.A * y

    def exact(self, t):             #Probably can't handle matrices
        return self.y0 * scipy.linalg.expm(t*self.A)

class Solver:
    def __init__(self, ivp, N, tol = 1e-6, adaptive = bool):
        self.ivp = ivp
        self.N = N
        self.u = ivp.y0
        if adaptive == False:
            self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
        else: 
            self.grid = np.zeros(0)
        self.h = (ivp.T - ivp.t0) / N
        self.tol = tol
        self.adaptive = adaptive

    def step(self, tn, un):
        raise NotImplementedError

class Evaluator:                               #Evaluates error for solver class when exact answer is known
    def __init__(self, ivp, solver_cls):
        self.ivp = ivp
        self.solver = solver_cls
      
    def errvstime(self, N, bool):               # Returns a tuple of t values and corresponding cumulative error
        errs = np.zeros(N + 1)                  # If bool = true returns relative error, else absolute error
        solver = self.solver(self.ivp, N)  
        ivp = self.ivp                                   #Initiating solver inside method
        h = solver.h
        for i in range(N):
            errs[i] = np.linalg.norm(solver.u - ivp.exact(h * i))
            solver.u = solver.step(h * i, solver.u)
        errs[N] = np.linalg.norm(solver.u - ivp.exact(h * N))
        if bool == False:
            return solver.grid, errs                               
        else: 
            for i in range(len(errs)):
                errs[i] = errs[i] / np.linalg.norm(ivp.exact(h*i))
            return solver.grid, errs

    def errvsh(self, Ns, bool):         # Takes array of different N and 
        final_errs = np.zeros(0)        # returns tuple of 1. global error at T for different N 2. grid sizes h for respective N
        grid_sizes = np.zeros(0)        # If bool = true returns absolute error, else relative error
        for N in Ns:
            solver = self.solver(self.ivp, N)
            ts, us = solver.integrate()
            final_u = us[len(us) - 1]
            final_y = self.ivp.exact(self.ivp.T)
            final_errs = np.append(final_errs, np.linalg.norm(final_u - final_y))
            grid_sizes = np.append(grid_sizes, solver.h)
        if bool == False:        
            return final_errs, grid_sizes
        if bool == True:
            new_errs = np.zeros(0)
            for err in final_errs:
                err = err / np.linalg.norm(final_y)
                new_errs = np.append(new_errs, err)
            return new_errs, grid_sizes
    