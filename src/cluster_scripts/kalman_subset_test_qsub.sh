#!/bin/bash

#Torque directives
#PBS -N seq-conn-brute-force-small
#PBS -W group_list=hpcstats
#PBS -l nodes=1:ppn=8,walltime=24:00:00,mem=2000mb
#PBS -M bms2156@columbia.edu
#PBS -m abe
#PBS -V

#set output and error directories
#PBS -o localhost:/hpc/stats/users/bms2156/log/
#PBS -e localhost:/hpc/stats/users/bms2156/log/


# run directory (top-level dir)
run_dir="/hpc/stats/users/bms2156/common-input/"
# repo dir
repo_dir="${run_dir}common-input/"

# number of timesteps to run
T="100000"
# number of neurons total
N="100"
# number of neurons in each observed subset
K="20"
# pick a seed for the random number generator
rng_seed="84720"
# output path
CUR_TIME=`date +"%d%m%y-%H%M"`
output_path="${run_dir}data/kalman_subset_test_${K}_of_${N}_${CUR_TIME}.mat"

# job command to feed to matlab
job="kalman_subset_cluster(${N}, ${K}, ${T}, '${output_path}', ${repo_dir}, ${rng_seed})"

# options for matlab executable
matlab_opts="-nosplash -nodisplay -nodesktop "

#run matlab executable
echo "$job" | \
    MATLABPATH="${repo_dir}src" \
    matlab $matlab_opts \
    1> "${run_dir}log/brute-force-${NAME}-${num_neurons}-of-${K}.matlab.out" \
    2> "${run_dir}log/brute-force-${NAME}-${num_neurons}-of-${K}.matlab.err"

#End of script

