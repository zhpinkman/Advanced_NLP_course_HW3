import os
import subprocess


all_models = os.listdir('.')
all_models = [model for model in all_models if model.startswith(
    'model.') or model == 'train.model']


for model in all_models:
    print(model)
    output = subprocess.run(['python', 'parse.py', '-m', model, '-i',
                            'dev.orig.conll', '-o', 'dev.parse.out'])

    output = subprocess.run(['java', '-cp', 'stanford-parser.jar', 'edu.stanford.nlp.trees.DependencyScoring',
                            '-g', 'dev.orig.conll', '-conllx', 'True', '-s', 'dev.parse.out'])
