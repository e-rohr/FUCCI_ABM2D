#!/bin/bash
        
#BATCH --chdir=./                     # Set the working directory
#SBATCH --array=0-62
#SBATCH --mail-user=email@website.com      # Who to send emails to
#SBATCH --mail-type=ALL                  # Send emails on start, end and failure
#SBATCH --job-name=soboldata_gen       # Name to show in the job queue
#SBATCH --output=out_files/job.soboldata_gen.%a.out              # Name of stdout output file (%a expands to SLURM_ARRAY_TASK_ID)
#SBATCH --ntasks-per-node=1                       # Total number of mpi tasks requested
#SBATCH --nodes=1                        # Total number of nodes requested
#SBATCH --partition=long                 # Partition (a.k.a. queue) to use
#SBATCH --time=2-00:00:00                # Max run time (days-hh:mm:ss) ... adjust as necessary

source activate FUCCI_ABM2D

# Run serial program
echo "Starting on "`date`
python sobol_data_generation.py "$SLURM_ARRAY_TASK_ID"
echo "Finished on "`date`
