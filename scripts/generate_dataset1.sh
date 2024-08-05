#!/bin/bash
        
#BATCH --chdir=./                     # Set the working directory
#SBATCH --array=0-120
#SBATCH --mail-user=rohre1@tcnj.edu      # Who to send emails to
#SBATCH --mail-type=ALL                  # Send emails on start, end and failure
#SBATCH --job-name=dataset1_gen       # Name to show in the job queue
#SBATCH --output=job.dataset1_gen.%j.out              # Name of stdout output file (%j expands to jobId)
#SBATCH --ntasks-per-node=1                       # Total number of mpi tasks requested
#SBATCH --nodes=1                        # Total number of nodes requested
#SBATCH --partition=short                 # Partition (a.k.a. queue) to use
#SBATCH --time=0-01:00:00                # Max run time (days-hh:mm:ss) ... adjust as necessary

module add python/3.10.11

# Run serial program
echo "Starting on "`date`
python data_generation.py "1" "$SLURM_ARRAY_TASK_ID"
echo "Finished on "`date`