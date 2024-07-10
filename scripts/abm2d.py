import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import scipy.sparse as sp
import time
import itertools
import multiprocessing 
import Cabm2d
import Ccell_dens






####################################################################################################
##################################### Subroutines ##################################################


def unif(R,L,rng):
    r = R * np.sqrt(rng.uniform())
    θ = rng.uniform() * 2*np.pi
    x = L/2.0 + r*np.cos(θ)
    y = L/2.0 + r*np.sin(θ)

    return (x,y)

####################################################################################################
    
def visualize_cell(X, Y, state, xmesh,ymesh, c_square, t):
    fontsize = 20
    fig, ax  = plt.subplots(1,2, figsize=(18,8), gridspec_kw={'width_ratios': [1, 1.3]})
    colormap = np.array(["lightseagreen", "red", "yellow", "lime"])
    ax[0].scatter(X,Y,c=colormap[state])
    ax[0].set_xlabel("$x$",fontsize=fontsize)
    ax[0].set_ylabel("$y$",fontsize=fontsize)
    ax[0].set_title(f"2D Tumor Spheroid at $t$ = {t}",fontsize=fontsize)
    ax[1].contourf(xmesh, ymesh, c_square, levels=10)
    plt.colorbar(plt.contourf(xmesh, ymesh, c_square, levels=10), ax = ax[1], label = "$C(x,y)$")
    ax[1].set_xlabel("$x$",fontsize=fontsize)
    ax[1].set_ylabel("$y$",fontsize=fontsize)
    ax[1].set_title(f"Nutrient Concentration Profile at $t$ = {t}",fontsize=fontsize)
    fig.tight_layout()
    
    plt.show()
    return    

####################################################################################################
######################################### Main Routine #############################################

def abm2d(path = "../data/",title = None, iter = -1, N = 2500,L = 1000,T = 240,I = 201,α = 0.0075,dmax = 2,dmin = 0.0005,mmax = .12,mmin = .06,Rr = .047,Ry = .4898,Rg = .0619, η1=5, η2=5, η3=15, c_a =0.4, c_d=0.1,c_m=0.5):
    ####################################################################################################
    ##################################### Parameters ###################################################
    rng = np.random.default_rng()

    # Numerical Parameters
    Nmax = 30000
    #N = 2000                          # Initial number of cells
    Nd  = 0                           # Initial number of dead cells
    #L = 1000                          # Side Length of Domain   (μm)
    #T = 240                           # Maximum Simulation Time (hours)                                                           
    r_o = 245                         # initial spheroid radius  (μm)
    μ = 12                            # Migration distance       (μm)
    σ = 12                            # Dispersal distance after mitosis (μm)
    #I = 101

    # Nutrient Parameters
    #c_d = 0.1                          # Critical death concentration
    #c_a = 0.4                          # Critical arrest concentration
    #c_m = 0.5                          # Critical migration concentration


    # Per Capita Agent Rates
    #dmax = 2                          # Maximum death rate
    #dmin = 0.0005                     # Minimum death rate
    #mmax = 0.12                       # Maximum migration rate
    #mmin = 0.06                       # Minimum migration rate
    #η1 = 5                            # Hill function index for arrest
    #η2 = 5                            # Hill function index for migration
    #η3 = 15                           # Hill function index for death
    #Rr = 0.047                        # Red->Yellow transition rate
    #Ry = 0.4898                       # Yellow->Green transition rate
    #Rg = 0.0619                       # Green->Red transition rate (mitosis)



    # Finite Mesh Grid
    xgrid = np.linspace(0, L, I)
    ygrid = np.linspace(0, L, I)
    h = L/(I-1)
    C_b = 1                           # Boundary condition
    #α = .015                        # Consumption-Diffusion rate
    pdeT = 1

    # State Variables
    X = np.ones(Nmax) * (-1)
    Y = np.ones(Nmax) * (-1)
    state = np.ones(Nmax, dtype = int)*(-1)
    c_p = np.empty(Nmax)
    M = np.empty(Nmax)
    D = np.empty(Nmax)
    cycr = np.empty(Nmax)




    ################################## Equations (1), (4), and (5) #####################################

    Rr_c = lambda c : Rr*(c**η1 / (c_a**η1 + c**η1))                       # (1)

    m_c  = lambda c : (mmax - mmin)*(c**η2)/(c_m**η2 + c**η2) + mmin       # (4)

    d_c  = lambda c : (dmax - dmin)*(1 - (c**η3)/(c_d**η3 + c**η3)) + dmin # (5)



    # Initialize cell positions

    for q in range(N):
        X[q],Y[q] = unif(r_o,L,rng)

    # Approximate cell density
    C = np.zeros(I**2)
    Ccell_dens.celldens_demo_serial(I,xgrid,ygrid,X,Y,N,h, C)
    
    C_out = np.empty((T+1,I**2))
    C_out[0] = C
    

    
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
    data[4*I-4 + 4*(I**2 - 4*I + 4) : 4*I-4 + 5*(I**2 - 4*I + 4)] = [-C[KIJ(i,j,I)]*α - 4/h**2   for i in range(1,I-1) for j in range(1,I-1)]

    A = sp.csc_matrix((data, (row, col)), shape=(I**2, I**2))   
    c_vec, exitCode = sp.linalg.gmres(A, b, x0 = np.ones_like(b), tol = 1e-08)
    c_old = c_vec
    c_grid = np.reshape(c_vec, (I,I))






    # Initialize cell cycle states
    C_A_BASE = 0.4

    # Initialise cell type (red, yellow, green) storage - no initial dead cells
    Nr = 0; Ny = 0; Ng = 0 # Initialise counts of cells of each type

    # Find proportion of red/yellow/green in freely cycling conditions
    full_cycle = 1/Rr + 1/Ry + 1/Rg # Time of cell cycle -- Equation (S10)
    redprop = 1/Rr / full_cycle # Proportion of time spent in red
    yelprop = 1/Ry / full_cycle # Proportion of time spent in red
    greprop = 1/Rg / full_cycle # Proportion of time spent in red

    for q in range(N): #Interpolating nutrient concentration

        c_p[q] = Cabm2d.interp2d(I, h, X[q], Y[q], c_vec)

        if c_p[q] > C_A_BASE: # Identify cycling status of cell (region-wise)
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

        else: # Tuned to get an initial arrest radius matching 
            if (c_p[q] > C_A_BASE - 0.137): # Rr(c) < 0.01*Rr -- consider this as an initial arrest region. Testing shows this qualitatively matches the arrest regions in the spheroids with variable size.
                u = rng.uniform()
                if u < 0.16: # reduced chance of greens and yellows in arrest zone. Arrest calculated when green radial density drops below 20%
                    u2 = rng.uniform()
                    if u2 < (1/Ry)*(1/Ry + 1/Rg): # Time in yellow / time in yellow and green
                        state[q] = 2
                        Ny += 1
                    else:
                        state[q] = 3
                        Ng += 1
                else: # (1/Rg)/(1/Ry + 1/Rg) condition: time in green / time in yellow and green
                    state[q] = 1
                    Nr += 1
            else: # All red when local concentration under c_a - 0.137
                state[q] = 1
                Nr += 1

        # Initialize cell rates (cycling, moving, death)
        if (state[q] == 1):
            cycr[q] = Rr_c(c_p[q]) # Equation (1)
        elif (state[q] == 2):
            cycr[q] = Ry # Equation (2)
        elif (state[q] == 3):
            cycr[q] = Rg # Equation (3)

        M[q] = m_c(c_p[q]) # Equation (4)
        D[q] = d_c(c_p[q]) # Equation (5)



    

    cellN = np.array([Nr,Ny,Ng,N,Nd])
    rates = np.array([dmax, dmin, Rr, Ry, Rg, mmax, mmin])
    hyp = np.array([c_a, c_m, c_d])
    
    cellN_out = np.empty((5,T+1))
    cellN_out[0,0] = Nr
    cellN_out[1,0] = Ny
    cellN_out[2,0] = Ng
    cellN_out[3,0] = N
    cellN_out[4,0] = Nd
    
    X_out = np.empty((T+1,Nmax))
    Y_out = np.empty((T+1,Nmax))
    state_out = np.empty((T+1,Nmax))
    X_out[0] = X
    Y_out[0] = Y
    state_out[0] = state
    
    count = 0
    t = 0
    while t < T:

        # Simulate Cell Events
        (cellN, rates, hyp, Nmax, X, Y, state, D, M, cycr, c_p, c_vec, C, μ, σ, h, pdeT, η1, η2, η3, I) = Cabm2d.cells_gillespie(cellN, rates, hyp, Nmax, X, Y, state, D, M, cycr, c_p, c_vec, C, μ, σ, h, pdeT, η1, η2, η3, I, rng)
        
        
        # Update nutrient concentration
        data[4*I-4 + 4*(I**2 - 4*I + 4) : 4*I-4 + 5*(I**2 - 4*I + 4)] = [-C[KIJ(i,j,I)]*α - 4/h**2   for i in range(1,I-1) for j in range(1,I-1)]
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
        C_out[count] = C
        X_out[count] = X
        Y_out[count] = Y
        state_out[count] = state
    
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
    
    out['C'] = C_out
    out['c_grid'] = c_grid
    out['T'] = T
    out['I'] = I
    out['α'] = α
    out['L'] = L
    
    out['Nr'] = cellN_out[0]
    out['Ny'] = cellN_out[1]
    out['Ng'] = cellN_out[2]
    
    out['η1'] = η1
    out['η2'] = η2
    out['η3'] = η3
    
    out['c_a'] = c_a
    out['c_d'] = c_d
    out['c_m'] = c_m
    
    if (title != None):
        np.save(f"{path}/{title}.npy",out,allow_pickle=True)
    else:
        if (iter > -1):
            np.save(f"{path}/abm2d iter {iter} Rr {Rr} Ry {Ry} Rg {Rg} dmax {dmax} mmax {mmax} η1 {η1} η2 {η2} η3 {η3} c_a {c_a} c_d {c_d} c_m {c_m}.npy",out,allow_pickle=True)
        else:
            np.save(f"{path}/abm2d Rr {Rr} Ry {Ry} Rg {Rg} dmax {dmax} mmax {mmax} η1 {η1} η2 {η2} η3 {η3} c_a {c_a} c_d {c_d} c_m {c_m}.npy",out,allow_pickle=True)