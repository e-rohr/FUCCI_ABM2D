import SALib
from SALib.sample import saltelli
# from SALib.analyze import sobol
# from SALib.test_functions import Ishigami
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from src.abm2d import abm2d

if __name__ == "__main__":
    
    hpc_index = int(sys.argv[1])
    
    problem = {
    'num_vars': 10,
    'names': ['c_a', 'c_d', 'c_m', 'dmax', 'dmin', 
              'mmax', 'mmin', 'eta1', 'eta2', 'eta3'],
    'bounds': [ [0.5*0.4,    2.0*0.4],
                [0.5*0.1,    2.0*0.1],
                [0.5*0.5,    2.0*0.5],
                [0.5*2,      2.0*2],
                [0.5*0.0005, 2.0*0.0005],
                [0.5*0.12,   2.0*0.12],
                [0.5*0.06,   2.0*0.06],
                [0.5*5,      2.0*5],
                [0.5*5,      2.0*5],
                [0.5*15,     2.0*15]]
    }

    param_values = saltelli.sample(problem, calc_second_order=False, N=512)
    
    for i in range(100):
        
        sample_index = hpc_index*100 + i
        
        if sample_index < len(param_values):
        
            p = param_values[sample_index]

            success_count = 0
            max_attempts = 100
            attempts = 0

            while success_count < 10 and attempts < max_attempts:
                try:
                    args = {'c_a':p[0], 
                        'c_d':p[1], 
                        'c_m':p[2], 
                        'dmax':p[3], 
                        'dmin':p[4], 
                        'mmax':p[5], 
                        'mmin':p[6], 
                        'eta1':p[7], 
                        'eta2':p[8], 
                        'eta3':p[9],
                        "path": f"../data/sobol/sample_{sample_index}",
                        "title": f"iter_{success_count}"}

                    if not os.path.isdir(args['path']):
                        os.makedirs(args["path"])
                        
                    abm2d(**args)
                    
                    success_count += 1
                    
                except Exception as e:
                    
                    print(f"Failure for sample {sample_index}, attempt: {attempts}: {e}")
                
                attempts += 1
        