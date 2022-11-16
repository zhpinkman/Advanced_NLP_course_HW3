#!/bin/bash
#SBATCH --job-name=dep_parse
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/dep_parsing/dependency-parsing
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate general


# python train.py

# python preparedata.py --mode train 

# python preparedata.py --mode dev


python train.py -m model.without_word_embedding --random_word_embedding

python parse.py -i dev.orig.conll -o dev.parse.out -m model.without_word_embedding

java -cp stanford-parser.jar edu.stanford.nlp.trees.DependencyScoring -g dev.orig.conll -conllx True -s dev.parse.out

conda deactivate