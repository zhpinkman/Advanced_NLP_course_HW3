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



# python preparedata.py --mode train 

# python preparedata.py --mode dev

# python train.py -m train.model

# python train.py -m model.without_word_embedding --random_word_embedding

# python train.py -m model.more_epochs --epochs 10

# python train.py -m model.optimizer_adam --optimizer adam

# python train.py -m model.data_ratio_0.1 --data_ratio 0.1

# python train.py -m model.data_ratio_0.4 --data_ratio 0.4

# python train.py -m model.data_ratio_0.7 --data_ratio 0.7

# python train.py -m model.n_features_100 --n_features 100

# python train.py -m model.n_features_200 --n_features 200

# python train.py -m model.n_features_300 --n_features 300

# python train.py -m model.hidden_dim_500 --hidden_dim 500

# python train.py -m model.activation_function_tanh --activation_function tanh

# python train.py -m model.use_dropout --use_dropout --epochs 10

# python train.py -m model.hidden_dim_50 --hidden_dim 50

# python train.py -m model.hidden_dim_100 --hidden_dim 100

# python parse.py -i dev.orig.conll -o dev.parse.out -m model.without_word_embedding

# java -cp stanford-parser.jar edu.stanford.nlp.trees.DependencyScoring -g dev.orig.conll -conllx True -s dev.parse.out

conda deactivate