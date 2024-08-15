import numpy as np
import math
import scipy.sparse as sp
import sys, importlib

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()},
                  language_level = 3
                 )
import src.cells as cells


####################################################################################################
##################################### Subroutine ##################################################


def unif(R,L,rng):
    r = R * np.sqrt(rng.uniform())
    θ = rng.uniform() * 2*np.pi
    x = L/2.0 + r*np.cos(θ)
    y = L/2.0 + r*np.sin(θ)

    return (x,y)

####################################################################################################
######################################### Main Routine #############################################

'''
Main routine: Runs ABM and outputs simulation results.

Numerical Parameters:
Nmax    - Maximum # of agents
N       - Total # of living agents
Nd      - # of dead agents
L       - Domain side length
T       - Total simulation time steps (in hrs)
r_o     - Initial spheroid radius
μ       - Agent migration distance
σ       - Daughter agent dispersal distance (same as μ)
I       - # of nodes along each axis (total # of nodes is I^2)
α       - Consumption/Diffusion ratio

Nutrient Parameters
c_d     - Critical death concentration
c_a     - Critical arrest concentration
c_m     - Critical migration concentration

Per Capita Agent Rates
dmax    - Maximum death rate
dmin    - Minimum death rate
mmax    - Maximum migration rate
mmin    - Minimum migration rate
eta1      - Hill function index for arrest
eta2      - Hill function index for migration
eta3      - Hill function index for death
Rr      - Red->Yellow transition rate
Ry      - Yellow->Green transition rate
Rg      - Green->Red transition rate (mitosis)
'''
def abm2d(path = "../data/", title = None,
          Nmax = 30000, N = 1100, L = 1000, T = 240, 
          I = 200, α = 0.01, pdeT = 1,
          dmax = 2, dmin = 0.0005, mmax = .12, mmin = .06,
          Rr = .047, Ry = .4898, Rg = .0619,
          eta1=5, eta2=5, eta3=15, c_a =0.4, c_d=0.1, c_m=0.5,
          r_o = 245, μ = 12, σ = 12 ):
    ####################################################################################################
    ##################################### Parameters ###################################################
    rng = np.random.default_rng()

    Nd  = 0                    # Initial number of dead cells                                                    
    n_days = int(T/24)

    # Finite Mesh Grid
    h = L/(I-1)
    C_b = 1                    # Boundary condition

    # State Variables
    X = np.full(shape = Nmax, fill_value = -1, dtype = np.float64)
    Y = np.full(shape = Nmax, fill_value = -1, dtype = np.float64)
    state = np.full(shape = Nmax, fill_value = -1, dtype = int)
    c_p = np.empty(Nmax)
    M = np.empty(Nmax)
    D = np.empty(Nmax)
    cycr = np.empty(Nmax)




    ################################## Equations (1), (4), and (5) #####################################

    Rr_c = lambda c : Rr*(c**eta1 / (c_a**eta1 + c**eta1))                       # (1)

    m_c  = lambda c : (mmax - mmin)*(c**eta2)/(c_m**eta2 + c**eta2) + mmin       # (4)

    d_c  = lambda c : (dmax - dmin)*(1 - (c**eta3)/(c_d**eta3 + c**eta3)) + dmin # (5)



    # Initialize cell positions

    for q in range(N):
        X[q],Y[q] = unif(r_o,L,rng)

    # Approximate cell density
    v = np.zeros(I**2)
    cells.density(I,X,Y,N,h,v)
    
    

    
    # Solve Equation (S8) for initial nutrient concentration profile

    b = np.zeros(I**2) 

    # Indices for vectorizing 2D domain

    KIJ   = lambda i,j,I : j*I + i
    KIM1J = lambda i,j,I : j*I + i - 1
    KIP1J = lambda i,j,I : j*I + i + 1
    KIJM1 = lambda i,j,I : (j-1)*I + i
    KIJP1 = lambda i,j,I : (j+1)*I + i

    # Construct coefficient matrix in Compressed Sparse Column (CSC) format
    row  = np.empty(4*I-4 + 5*(I**2 - 4*I + 4))
    col  = np.empty(4*I-4 + 5*(I**2 - 4*I + 4))
    data =  np.ones(4*I-4 + 5*(I**2 - 4*I + 4))

    # left boundary
    row[:I] = range(0, I**2-I+1, I)
    col[:I] = range(0, I**2-I+1, I)
    b[0: I**2-I+1: I] = C_b

    # right boundary
    row[I:2*I] = range(I-1,I**2, I)
    col[I:2*I] = range(I-1,I**2, I)
    b[I-1:I**2:I] = C_b

    # bottom boundary (excluding overlap with left/right boundary)
    row[2*I:3*I-2] = range(1, I-1)
    col[2*I:3*I-2] = range(1, I-1)
    b[1:I-1] = C_b

    # top boundary (excluding overlap with left/right boundary)
    row[3*I-2:4*I-4] = range(I**2 - I + 1,I**2 - 1)
    col[3*I-2:4*I-4] = range(I**2 - I + 1,I**2 - 1)
    b[I**2 - I + 1:I**2 - 1] = C_b

    # interior
    row[4*I-4  : 4*I-4 + (I**2 - 4*I + 4)] = [KIJ(i,j,I)   for i in range(1,I-1) for j in range(1,I-1)]
    col[4*I-4  : 4*I-4 + (I**2 - 4*I + 4)] = [KIM1J(i,j,I) for i in range(1,I-1) for j in range(1,I-1)]
    data[4*I-4 : 4*I-4 + (I**2 - 4*I + 4)] /= h**2

    row[4*I-4 + (I**2 -  4*I + 4)  : 4*I-4 + 2*(I**2 - 4*I + 4)] = [KIJ(i,j,I)   for i in range(1,I-1) for j in range(1,I-1)]
    col[4*I-4 + (I**2 -  4*I + 4)  : 4*I-4 + 2*(I**2 - 4*I + 4)] = [KIP1J(i,j,I) for i in range(1,I-1) for j in range(1,I-1)]
    data[4*I-4 + (I**2 - 4*I + 4) : 4*I-4 + 2*(I**2 - 4*I + 4)] /= h**2

    row[4*I-4 + 2*(I**2 -  4*I + 4)  : 4*I-4 + 3*(I**2 - 4*I + 4)] = [KIJ(i,j,I)   for i in range(1,I-1) for j in range(1,I-1)]
    col[4*I-4 + 2*(I**2 -  4*I + 4)  : 4*I-4 + 3*(I**2 - 4*I + 4)] = [KIJM1(i,j,I) for i in range(1,I-1) for j in range(1,I-1)]
    data[4*I-4 + 2*(I**2 - 4*I + 4) : 4*I-4 + 3*(I**2 - 4*I + 4)] /= h**2

    row[4*I-4 + 3*(I**2 -  4*I + 4)  : 4*I-4 + 4*(I**2 - 4*I + 4)] = [KIJ(i,j,I)   for i in range(1,I-1) for j in range(1,I-1)]
    col[4*I-4 + 3*(I**2 -  4*I + 4)  : 4*I-4 + 4*(I**2 - 4*I + 4)] = [KIJP1(i,j,I) for i in range(1,I-1) for j in range(1,I-1)]
    data[4*I-4 + 3*(I**2 - 4*I + 4) : 4*I-4 + 4*(I**2 - 4*I + 4)] /= h**2

    row[4*I-4 + 4*(I**2 - 4*I + 4)  : 4*I-4 + 5*(I**2 - 4*I + 4)] = [KIJ(i,j,I)   for i in range(1,I-1) for j in range(1,I-1)]
    col[4*I-4 + 4*(I**2 - 4*I + 4)  : 4*I-4 + 5*(I**2 - 4*I + 4)] = [KIJ(i,j,I)   for i in range(1,I-1) for j in range(1,I-1)]
    data[4*I-4 + 4*(I**2 - 4*I + 4) : 4*I-4 + 5*(I**2 - 4*I + 4)] = [-v[KIJ(i,j,I)]*α - 4/h**2   for i in range(1,I-1) for j in range(1,I-1)]

    A = sp.csc_matrix((data, (row, col)), shape=(I**2, I**2))   
    c_vec, exitCode = sp.linalg.gmres(A, b, x0 = np.ones_like(b), tol = 1e-08)
    c_old = c_vec
    c_grid = np.reshape(c_vec, (I,I))






    # Initialize cell cycle states
    C_A_BASE = 0.4

    # Initialise cell type (red, yellow, green) storage - no initial dead cells
    Nr = 0; Ny = 0; Ng = 0 # Initialise counts of cells of each type

    # Find proportion of red/yellow/green in freely cycling conditions
    full_cycle = (1/Rr) + (1/Ry) + (1/Rg) # Time of cell cycle -- Equation (S10)
    redprop = (1/Rr) / full_cycle # Proportion of time spent in red
    yelprop = (1/Ry) / full_cycle # Proportion of time spent in red
    greprop = (1/Rg) / full_cycle # Proportion of time spent in red
    
    for q in range(N):
        c_p[q] = cells.interp2d(I,h,X[q],Y[q],c_vec)
        
        if c_p[q] > C_A_BASE:
            u = rng.uniform()
            
            if u < redprop:
                state[q] = 1
                Nr += 1
            elif (u < redprop + yelprop):
                state[q] = 2
                Ny += 1
            elif (u < redprop + yelprop + greprop):
                state[q] = 3
                Ng += 1
        else:
            if Rr*(c_p[q]**eta1 / (C_A_BASE**eta1 + c_p[q]**eta1)) > 0.1*Rr:
                u = rng.uniform()
                if u < 0.16:
                    u2 = rng.uniform()
                    if u2 < (1/Ry)/((1/Ry) + (1/Rg)):
                        state[q] = 2
                        Ny += 1
                    else:
                        state[q] = 3
                        Ng += 1
                else:
                    state[q] = 1
                    Nr += 1
            else:
                state[q] = 1
                Nr += 1
        # Initialize cell rates (cycling, moving, death)
        if (state[q] == 1):
            cycr[q] = Rr*(c_p[q]**eta1 / (C_A_BASE**eta1 + c_p[q]**eta1)) # Equation (1)
          
        elif (state[q] == 2):
            cycr[q] = Ry # Equation (2)
          
        elif (state[q] == 3):
            cycr[q] = Rg # Equation (3)
          
        M[q] = m_c(c_p[q]) # Equation (4)
        D[q] = d_c(c_p[q]) # Equation (5)


    

    cellN = np.array([Nr,Ny,Ng,N,Nd])
    rates = np.array([dmax, dmin, Rr, Ry, Rg, mmax, mmin])
    hyp = np.array([c_a, c_m, c_d])
    
    # Record cell counts at each hour
    cellN_out = np.empty((5,T+1))
    cellN_out[0,0] = Nr
    cellN_out[1,0] = Ny
    cellN_out[2,0] = Ng
    cellN_out[3,0] = N
    cellN_out[4,0] = Nd
    
    # Record agent positions at each day
    X_out = np.empty((n_days + 1,Nmax))
    Y_out = np.empty((n_days + 1,Nmax))
    state_out = np.empty((n_days + 1,Nmax))
    X_out[0] = X
    Y_out[0] = Y
    state_out[0] = state
    
    count = 0
    t = 0
    while t < T:

        # Simulate Cell Events
        (cellN, X, Y, state, D, M, cycr, c_p, c_vec, v) = cells.gillespie(cellN, rates, hyp,
                                                                          Nmax, X, Y, state,
                                                                          D, M, cycr,
                                                                          c_p, c_vec, v,
                                                                          μ, σ, h, pdeT,
                                                                          eta1, eta2, eta3,
                                                                          I, rng)
        
        
        # Update nutrient concentration
        data[4*I-4 + 4*(I**2 - 4*I + 4) : 4*I-4 + 5*(I**2 - 4*I + 4)] = [-v[KIJ(i,j,I)]*α - 4/h**2   for i in range(1,I-1) for j in range(1,I-1)]
        A = sp.csc_matrix((data, (row, col)), shape=(I**2, I**2))   
        c_vec, exitCode = sp.linalg.gmres(A, b, x0 = c_old, tol = 1e-06)
        c_grid = np.reshape(c_vec, (I,I))
        c_old = c_vec
        
        N = cellN[3]
        Nd = cellN[4]
    

        t += pdeT
        count += 1
        
        
        cellN_out[0,count] = cellN[0]
        cellN_out[1,count] = cellN[1]
        cellN_out[2,count] = cellN[2]
        cellN_out[3,count] = cellN[3]
        cellN_out[4,count] = cellN[4]
        
        # record agent positions and state every 24hrs
        if t % 24 == 0:
            day = t // 24
            X_out[day] = X
            Y_out[day] = Y
            state_out[day] = state
            
    out = {}
    
    
    out['dmax'] = dmax
    out['dmin'] = dmin
    out['mmax'] = mmax
    out['mmin'] = mmin
    
    out['Rr'] = Rr
    out['Ry'] = Ry
    out['Rg'] = Rg
    
    out['X'] = X_out[:,:N+Nd]
    out['Y'] = Y_out[:,:N+Nd]
    out['state'] = state_out[:,:N+Nd]
    out['N'] = cellN_out[3]
    out['Nd'] = cellN_out[4]
    
    out['v'] = v
    out['c_grid'] = c_grid
    out['T'] = T
    out['I'] = I
    out['α'] = α
    out['L'] = L
    
    out['Nr'] = cellN_out[0]
    out['Ny'] = cellN_out[1]
    out['Ng'] = cellN_out[2]
    
    out['eta1'] = eta1
    out['eta2'] = eta2
    out['eta3'] = eta3
    
    out['c_a'] = c_a
    out['c_d'] = c_d
    out['c_m'] = c_m
    
    if (title != None):
        np.savez_compressed(f"{path}/{title}",**out)
    else:
        np.savez_compressed(f"{path}/abm2d_out",**out)