#!/bin/sh
#
# Specify the name for your job, this is the job name by which Slurm will refer to your job
# This can be different from the name of your executable or the name of your script file
#SBATCH --job-name Train
#SBATCH --partition  gpuq          # partition (queue)
# N = number of nodes - Only needed for parallel jobs
#SBATCH --nodes 1
## n= number of cores per node - You need to specify this in particular for parallel jobs
#SBATCH --ntasks-per-node 16
#SBATCH --mem=34G
#SBATCH --gres=gpu:4
# time needed to complete your job , note your job will preempt if it exceeds the time specified here
# Default is to combine output and error into a single file.  Next 2 options change that behavior
## Replace UserID with your Argo userID
#SBATCH --output /scratch/psanthal/slurm.%N.%j.out       # Output file
#SBATCH --error /scratch/psanthal/slurm.%N.%j.err        # Error file
#SBATCH --mail-type=ALL       # notifications for job state, XXX= END,FAIL  etc.
#SBATCH --mail-user=psanthal@gmu.edu       # send-to address
##SBATCH --time=00-12:00
# Load the relevant modules needed for the job
#module load python/python3.6.4
## Start the job
#python3 bin_lstm.py
#python3 spt_lstm.py
#python3 bin.py
#python3 gen.py
#python3 comp.py
#python3 embed.py
#python3 temp.py
#python3 multiView-V4.py
python3 word_sep_test.py


