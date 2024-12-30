#!/bin/bash

# SLURM job parameters
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
MAIL_USER="your_email@example.com"
MAIL_TYPE=END  # or BEGIN, FAIL, ALL
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

# Experiment parameters
K_VALUES=(32 64)
L_VALUES=(2 4 8 16)

# Counter to skip the first 5 jobs
COUNTER=0
SKIP_COUNT=5

for K in "${K_VALUES[@]}"; do
  for L in "${L_VALUES[@]}"; do
    COUNTER=$((COUNTER + 1))
    if (( COUNTER <= SKIP_COUNT )); then
      echo "Skipping job $COUNTER (K=$K, L=$L)"
      continue
    fi
    JOB_NAME="cnn_experiment_K${K}_L${L}"
    RUN_NAME="exp1_1"

    sbatch \
      -N $NUM_NODES \
      -c $NUM_CORES \
      --gres=gpu:$NUM_GPUS \
      --job-name $JOB_NAME \
      --mail-user $MAIL_USER \
      --mail-type $MAIL_TYPE \
      -o slurm-%j.out <<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Load conda environment
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run the Python experiment script
python -m hw2.experiments run-exp \
  -n $RUN_NAME \
  -K $K \
  -L $L \
  -P 4\
  -H 100

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

  done
done

