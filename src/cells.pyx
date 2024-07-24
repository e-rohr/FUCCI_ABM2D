import cython
import numpy as np
import math

cimport cython
from libc.math cimport fabs
from libc.math cimport round
cimport numpy as cnp

cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

###############################################################################
################################# Subroutines #################################

'''
Bilinear interpolation using the weighted mean formula at:

https://en.wikipedia.org/wiki/Bilinear_interpolation
'''
@cython.boundscheck(False)
@cython.wraparound(False)
def interp2d(int I, float h, float Xp, float Yp, cnp.ndarray[DTYPE_t, ndim = 1] v):
    
    # Find index location of agent
    cdef int i = math.ceil(Xp/h)
    cdef int j = math.ceil(Yp/h)

    # Find surrounding x y locations
    cdef float x1 = h*(i-1)
    cdef float x2 = h*i
    cdef float y1 = h*(j-1)
    cdef float y2 = h*j

    # Find indices and values of surrounding nutrient concentrations                             
    cdef int ind11 = I*(i-1) + j - 1                       
    cdef int ind21 = I*i + j - 1                           
    cdef int ind12 = I*(i-1) + j                           
    cdef int ind22 = I*i + j                               
    cdef float c11 = v[ind11]                                  
    cdef float c21 = v[ind21]                                  
    cdef float c12 = v[ind12]                                  
    cdef float c22 = v[ind22]                                  

    # Calculate 2D interpolation coefficients
    cdef float w11 = (x2 - Xp) * (y2 - Yp) / ((x2 - x1) * (y2 - y1))
    cdef float w12 = (x2 - Xp) * (Yp - y1) / ((x2 - x1) * (y2 - y1))
    cdef float w21 = (Xp - x1) * (y2 - Yp) / ((x2 - x1) * (y2 - y1))
    cdef float w22 = (Xp - x1) * (Yp - y1) / ((x2 - x1) * (y2 - y1))

    
    # Interpolate to agent location
    cdef float c_p = w11*c11 + w12*c12 + w21*c21 + w22*c22 
    
    return c_p

###############################################################################

'''
Approximate cell density by computing number of cells in each control area.

Modifies C vector inplace.
'''
@cython.boundscheck(False)
@cython.wraparound(False)
def density(int I,cnp.ndarray[DTYPE_t, ndim = 1] X,cnp.ndarray[DTYPE_t, ndim = 1] Y,int N,float h, cnp.ndarray[DTYPE_t, ndim = 1] v):
    cdef int i, j, index, q
    cdef float xag, yag
    
    # Iterate over each cell
    for q in range(N):
        xag = X[q]
        yag = Y[q]
        
        # Compute the index of the control area for the cell
        i = int(round(xag/h))
        j = int(round(yag/h))
        index = i*I + j
        
        # Update cell count for the control area
        v[index] += 1
    
    # Divide all cell counts by area h^2
    v /= h**2
    
    return

###############################################################################

# Equations (1,4,5) governing agent behavior rates

# Rate of committing to cycling
cdef inline Rr_c(float c, float eta1, float Rr, float c_a):               # (1)
    return Rr*(c**eta1 / (c_a**eta1 + c**eta1))

# Rate of migration
cdef inline m_c(float c, float eta2, float mmax, float mmin, float c_m):  # (4)
    return (mmax - mmin)*(c**eta2)/(c_m**eta2 + c**eta2) + mmin

# Rate of death
cdef inline d_c(float c, float eta3, float dmax, float dmin, float c_d):  # (5)
    return (dmax - dmin)*(1 - (c**eta3)/(c_d**eta3 + c**eta3)) + dmin

###############################################################################
############################### Main Routine ##################################
###############################################################################

'''
Runs Gillespie algorithm to resolve cellular events.
'''

@cython.boundscheck(False)
@cython.wraparound(False)
def gillespie(cnp.ndarray[cnp.int64_t, ndim = 1] cellN,
              cnp.ndarray[DTYPE_t, ndim = 1] rates,
              cnp.ndarray[DTYPE_t, ndim = 1] hyp,
              int Nmax,
              cnp.ndarray[DTYPE_t, ndim = 1] X,
              cnp.ndarray[DTYPE_t, ndim = 1] Y,
              cnp.ndarray[cnp.int64_t, ndim = 1] state,
              cnp.ndarray[DTYPE_t, ndim = 1] D,
              cnp.ndarray[DTYPE_t, ndim = 1] M,
              cnp.ndarray[DTYPE_t, ndim = 1] cycr,
              cnp.ndarray[DTYPE_t, ndim = 1] c_p,
              cnp.ndarray[DTYPE_t, ndim = 1] c,
              cnp.ndarray[DTYPE_t, ndim = 1] v,
              float mu,
              float sigma,
              float h,
              float pdeT,
              float eta1,
              float eta2,
              float eta3,
              int I,
              rng):


    
    
    # Extract values from input arrays
    cdef int Nr,Ny,Ng,N,Nd
    cdef float dmax, dmin, Rr, Ry, Rg, mmax, mmin, c_a, c_m, c_d
    
    # Define variable types for Cython
    cdef float dr, mr, cr, l, tau, u
    cdef int q, i, j, ind
    cdef float xp, yp, theta, x1, y1, x2, y2, xold, yold, xnew, ynew
    
    cdef int iold, jold, 
    cdef int inew1, jnew1, newind1, inew2, jnew2, newind2
    cdef int inew, jnew,
    cdef int newind, oldind


    (Nr,Ny,Ng,N,Nd) = cellN                        # Counts of each cell type and living/dead cells
    (dmax, dmin, Rr, Ry, Rg, mmax, mmin) = rates   # Per capita rates
    (c_a, c_m, c_d) = hyp
 
    cdef float t = 0
    
    while(t < pdeT and (N+Nd) < Nmax):
        
        # Calculate the individual event rates
        dr = np.sum(D[:N+Nd])
        mr = np.sum(M[:N+Nd]) 
        cr = np.sum(cycr[:N+Nd])
        
        # Calculate the total rate of events
        l = dr + mr + cr
        
        # Sample time step to next event tau ~ Exp(rate = l)
        tau = rng.exponential(1/l)
        t += tau

        # Break and output results without doing anything if t > pdeT
        if (t > pdeT):
            t = pdeT
            break

        # Choose an event
        u = l*rng.uniform()
    
############################### Cycling Event ######################################################
        
        if (u < cr):
            
            q = rng.choice(a = range(N+Nd), p = cycr[:N+Nd]/np.sum(cycr[:N+Nd]))
            
            # Red -> Yellow
            if (state[q] == 1):
                state[q] = 2
                Nr -= 1
                Ny += 1
                cycr[q] = Ry # Equation (2)
                
                xp = X[q]
                yp = Y[q]
                i = int(round(xp/h))
                j = int(round(yp/h))
                ind = I*i + j
            
            # Yellow -> Green
            elif(state[q] == 2):
                state[q] = 3
                Ny -= 1
                Ng += 1
                cycr[q] = Rg # Equation (3)
                
                xp = X[q]
                yp = Y[q]
                i = int(round(xp/h))
                j = int(round(yp/h))
                ind = I*i + j

                
            
            # Green -> 2 Reds (Mitosis)
            elif(state[q] == 3):
                Ng -= 1
                Nr += 2
                state[q] = 1
                xp = X[q]
                yp = Y[q]
                
                # Calculate the new positions of the cells
                theta = 2*np.pi*rng.uniform();
                
                x1 = xp + (sigma/2)*np.cos(theta)
                y1 = yp + (sigma/2)*np.sin(theta)
                
                x2 = xp - (sigma/2)*np.cos(theta)
                y2 = yp - (sigma/2)*np.sin(theta)
                
                X[q] = x1
                Y[q] = y1
        
                # Create new cell
                X[N+Nd] = x2
                Y[N+Nd] = y2
                state[N+Nd] = 1
                
                # Check nutrient concentrations and update agent rates
                
                c_p[q] = interp2d(I,h, x1, y1, c)
                
                cycr[q] = Rr_c(c_p[q], eta1, Rr, c_a) # Equation (1)
                M[q] = m_c(c_p[q], eta2, mmax, mmin, c_m) # Equation (4)
                D[q] = d_c(c_p[q], eta3, dmax, dmin, c_d) # Equation (5)
                
                
                c_p[N+Nd] = interp2d(I,h, x2, y2, c)
                
                cycr[N+Nd] = Rr_c(c_p[N+Nd], eta1, Rr, c_a) # Equation (1)
                M[N+Nd] = m_c(c_p[N+Nd], eta2, mmax, mmin, c_m) # Equation (4)
                D[N+Nd] = d_c(c_p[N+Nd], eta3, dmax, dmin, c_d) # Equation (5)
                
                # Account for increase in population
                N += 1
                
                # Update cell density
                
                iold = int(round(xp/h))
                jold = int(round(yp/h))
                oldind = I*iold + jold
                inew1 = int(round(x1/h))
                jnew1 = int(round(y1/h))
                newind1 = I*inew1 + jnew1
                
                
                if (oldind != newind1): # Cell moves from old volume to new one
                    v[oldind] -= 1/h**2
                    v[newind1] += 1/h**2
                    
                    
                
                inew2 = int(round(x2/h))
                jnew2 = int(round(y2/h))
                newind2 = I*inew2 + jnew2
                
                v[newind2] += 1/h**2 # One new cell in the finite volume surrounding node point
                
                v[oldind] = np.round(v[oldind], 4)
                v[newind1] = np.round(v[newind1], 4)
                v[newind2] = np.round(v[newind2], 4)

                
############################### Migration Event ####################################################

        elif (u < cr + mr):
            
            q = rng.choice(a = range(N+Nd), p = M[:N+Nd]/np.sum(M[:N+Nd]))
            
            # Sample random direction for cell to migrate towards
            theta = 2*np.pi*rng.uniform()
            xold = X[q]
            yold = Y[q]
            
            xnew = xold + mu*np.cos(theta)
            ynew = yold + mu*np.sin(theta)
            X[q] = xnew
            Y[q] = ynew
                
            # Update local nutrient concentration for agent, and rates
            c_p[q] = interp2d(I, h, xnew, ynew, c)
            
            M[q] = m_c(c_p[q], eta2, mmax, mmin, c_m) # Equation (4)
            D[q] = d_c(c_p[q], eta3, dmax, dmin, c_d) # Equation (5) 
            if (state[q] == 1):
                cycr[q] = Rr_c(c_p[q], eta1, Rr, c_a) # Equation (1) only update cycling rate if cell is red
                
            # Update cell density
            iold = int(round(xold/h))
            jold = int(round(yold/h))
            oldind = I*iold + jold
            
            inew = int(round(xnew/h))
            jnew = int(round(ynew/h))
            newind = I*inew + jnew
            if (oldind != newind): # Cell moves from old volume to new one
                v[oldind] -= 1/h**2
                v[newind] += 1/h**2
                
            v[oldind] = np.round(v[oldind], 4)
            v[newind] = np.round(v[newind], 4)

            
    
################################ Death Event #######################################################

        else:
            
            q = rng.choice(a = range(N+Nd), p = D[:N+Nd]/np.sum(D[:N+Nd]))
            xp = X[q]
            yp = Y[q]
            i = int(round(xp/h))
            j = int(round(yp/h))
            ind = I*i + j
            
            # Account for subpopulation counts and density changes
            if (state[q] == 1):
                Nr -= 1
            elif (state[q] == 2):
                Ny -= 1
            elif (state[q] == 3):
                Ng -= 1
                
            # "Kill" the cell
            state[q] = 0
            M[q] = 0
            D[q] = 0
            cycr[q] = 0
            
            # Account for population change
            N -= 1
            Nd += 1
            
            # Account for total cell density changes
            v[ind] -= 1/h**2
            
            v[ind] = np.round(v[ind], 4)

    
####################################### Update Parameters to Output ################################    
    
    cellN = np.array([Nr,Ny,Ng,N,Nd])
    rates = np.array([dmax, dmin, Rr, Ry, Rg, mmax, mmin])
    hyp = np.array([c_a, c_m, c_d])
    
    
    
    return (cellN, X, Y, state, D, M, cycr, c_p, c, v)
