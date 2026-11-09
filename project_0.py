import numpy as np
import scipy as sp
import scipy . linalg
import matplotlib . pyplot as plt

ivp = LTE ( t0 = 0 , T = 1 , y0 = np . array ([2]) , A = np . array
([[ -1]]) )
solver = ExplicitEuler ( ivp , N = 100)
solver . integrate ()
# Plot the approximations { u_n } vs . the temporal
# grid points { t_n }:
plt . plot ( solver . grid , solver . us )