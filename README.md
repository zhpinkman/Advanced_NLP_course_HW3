# dependency-parsing
Standard arc implementation of dependency parsing based on the paper "A Fast and Accurate Dependency Parser using Neural Networks". All the supplementary materials can be found [here](https://drive.google.com/file/d/1seNpXzd5eO8LfDiEBTxoCZj-zGZni2QX/view?usp=share_link).

The list of content: 

[Main Implementation](#main-implementation)

[Auxiliary Generated Models and Data Files](#auxiliary-generated-models-and-data-files)

[Structure of the Features Generted](#structure-of-the-features-generated)


<h1>


## Main Implementation

    There are three main parts of this implementation and also their corresponding implementation file.

1. Feature extraction and configuration generation `preparedata.py`

gets one argument which is the split of the data that has to be processed. 

2. Training a Dependency parser neural network based on the features and target dependency relations `train.py`

gets only one required argument which is the name of the model, and the rest of the arguments are essentially the hyper parameters that can change specific part of a model: 

`--train`: path to train file

`--dev`: path to dev file

`-m`: model name

`--random_word_embedding`: use random word embedding

`--optimizer`: optimizer to use

`--data_ratio`: ratio of data to use

`--epochs`:   'number of epochs

`--hidden_dim`, : 'hidden dimension of the fully connected layer

`--n_features`: number of features in the embedding layer

`--activation_function` :  activation function to use

`--use_dropout` :  use dropout


3. Greedy parsing of the sentences using the dependency parser from step 2 to get from the sentences to their dependency parse tree `parse.py`

gets the input file that has to be processes in the CoNLL structure, as well as the output file and the model to use as the dependency parser:

`-i`: input file

`-o`: output file

`-m`: model file

4. Evaluation of the generated trees based on Unlabeled / Labeled Attachment Score (UAS / LAS) for all the models generated `evaluate_models.py`

5. Script for running different stages of the whole framework on a GPU cluster with SLURM job scheduler `general.sh`

<h1>

## Auxiliary Generated Models and Data Files

    Also there are a couple of auxiliary files that are basically the   intermediary generated files or models: 

`dev.converted` & `train.converted`: the features extracted from the initial file for the purpose of training the dependency parser neural network ( with the classification task ) explained thoroughly at [Structure of the Features Generated](#structure-of-the-features-generated)

`train.parse.out` & `dev.parse.out`: the generated dependency parse trees in the conll structure for the train split and dev split

`model.*` & `train.model`: the trained dependency parser neural networks

`train.orig.conll` & `dev.orig.conll`: the raw data which are the sentences enriched with token level POS tags and dependency tree parent and dependency label

<h1>

## Structure of the Features Generated

`dev.converted` & `train.converted` are the features extracted from the initial file for the purpose of training the dependency parser neural network. 

Each line of these feature files represent one data point with data features with their target dependency label which is the action that has to be taken given the corresponding status of the stack and buffer in the configuration.

Each line has 49 entities separated by tab, the first 48 being the features and the last token being the target label that has to be predicted by the parser. In the 48 features, the first 18 features are word level features consist of: 

(1) the top three words in the buffer and stack $s_1, s_2, s_3, b_1, b_2, b_3$, 

(2) the first and second leftmost and rightmost children of the top two words on the stack $lc_1(s_i), rc_1(s_i) lc_2(s_i), rc_2(s_i), i = 1, 2$

(3) the leftmost of leftmost and rightmost of the rightmost children of the top two words on the stack $lc_1(lc_1(s_i)), rc_1(rc_1(s_i)), i = 1, 2$.

The second 18 features are the POS tags of the words extracted for the 18 word features, and the last 12 features are the dependency label of the last 12 words of the 18 words discussed in the word features, as there are no dependency labels for the words still in the stack or buffer and yet have to be processed to be assigned parent and dependency label.


####
