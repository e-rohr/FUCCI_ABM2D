#!/bin/bash
        
#BATCH --chdir=./                     # Set the working directory
#SBATCH --array=0-449
#SBATCH --mail-user=rohre1@tcnj.edu      # Who to send emails to
#SBATCH --mail-type=ALL                  # Send emails on start, end and failure
#SBATCH --job-name=dataset2_gen       # Name to show in the job queue
#SBATCH --output=job.dataset2_gen.%a.out              # Name of stdout output file (%a expands to SLURM_ARRAY_TASK_ID)
#SBATCH --ntasks-per-node=1                       # Total number of mpi tasks requested
#SBATCH --nodes=1                        # Total number of nodes requested
#SBATCH --partition=normal                 # Partition (a.k.a. queue) to use
#SBATCH --time=1-00:00:00                # Max run time (days-hh:mm:ss) ... adjust as necessary

module add python/3.10.11

# Run serial program
echo "Starting on "`date`
python data_generation.py "2" "$SLURM_ARRAY_TASK_ID"
echo "Finished on "`date`