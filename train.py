import torch
import pandas as pd
import numpy as np
import argparse
from IPython import embed
import os
import joblib
from tqdm import tqdm



def get_features_labels(file_name: str):
    df = pd.read_csv(file_name, sep='\t', header=None)
    features = df.iloc[:, 0:-1].values
    labels = df.iloc[:, -1].tolist()
    
    return features, labels
    
    
def get_all_unique_labels(features):
    file_name = 'unique_labels.joblib'
    if os.path.exists(file_name):
        print('Loading unique labels from file')
        unique_labels = joblib.load(file_name)
        return unique_labels
    label_features = features[:, 36:]
    unique_labels = np.unique(label_features)
    joblib.dump(unique_labels, file_name)
    return unique_labels
    
def get_all_unique_poss(features):
    file_name = 'unique_poss.joblib'
    if os.path.exists(file_name):
        print('Loading unique poss from file')
        unique_poss = joblib.load(file_name)
        return unique_poss
    poss_features = features[:, 18:36]
    unique_poss = np.unique(poss_features)
    joblib.dump(unique_poss, file_name)
    return unique_poss
    
def get_all_unique_words(features):
    file_name = 'unique_words.joblib'
    if os.path.exists(file_name):
        print('Loading unique words from file')
        unique_words = joblib.load(file_name)
        return unique_words
    word_features = features[:, 0:18]
    unique_words = np.unique(word_features)
    joblib.dump(unique_words, file_name)
    return unique_words


def create_random_embedding_matrix(vocab):
    word2id = {word: index for index, word in enumerate(vocab)}
    word2id['<UNK>'] = len(word2id)
    
    vocab_size=len(word2id)
    vector_size=300
    
    embedding_matrix=np.random.rand(vocab_size, vector_size)
    embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)
    embedding.weight = torch.nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
    return embedding, word2id

def create_word_embedding_matrix(vocab):
    glove = pd.read_csv('glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}    

    embedding_matrix=np.zeros((len(vocab), 300))
    
    word2id = dict()

    for index, word in enumerate(vocab):
        if word in glove_embedding:
            embedding_matrix[index]=glove_embedding[word]
        word2id[word] = index
        
    word2id['<UNK>'] = len(word2id)
    unk_matrix = np.zeros((1, 300))
    
    embedding_matrix = np.concatenate((embedding_matrix, unk_matrix), axis=0)
            
    vocab_size=embedding_matrix.shape[0]
    vector_size=embedding_matrix.shape[1]
 
    embedding = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=vector_size)
    embedding.weight = torch.nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
    return embedding, word2id

    
    
# class DependencyParser(torch.nn.Module):
#     def __init__(self, n_features: int, hidden_dim: int, num_labels: int) -> None:
#         super().__init__()
#         self.word_embeddings = 
        
        
class Tokenizer:
    def __init__(self, word2id, pos2id, label2id):
        self.word2id = word2id
        self.pos2id = pos2id
        self.label2id = label2id
    
    def tokenize_batch(self, features):
        return np.array([self.tokenize(sub_features) for sub_features in tqdm(features, leave = False)])
        
    def tokenize(self, features):
        results = []
        for word_features in features[:18]:
            for token in word_features:
                if token in self.word2id:
                    results.append(self.word2id[token])
                else:
                    results.append(self.word2id['<UNK>'])
        for pos_features in features[18:36]:
            for token in pos_features:
                if token in self.pos2id:
                    results.append(self.pos2id[token])
                else:
                    results.append(self.pos2id['<UNK>'])
        
        for label_features in features[36:]:
            for token in label_features:
                if token in self.label2id:
                    results.append(self.label2id[token])
                else:
                    results.append(self.label2id['<UNK>'])
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.oracle.txt")
    parser.add_argument("--dev", type=str, default="dev.oracle.txt")
    
    args = parser.parse_args()
    
    
    features, labels = get_features_labels(args.train)

    word_embedding, word2id = create_word_embedding_matrix(
        vocab=get_all_unique_words(features)        
    )
    print('loaded word embedding')
    pos_embedding, pos2id = create_random_embedding_matrix(
        vocab = get_all_unique_poss(features)
    )
    print('loaded pos embedding')
    label_embedding, label2id = create_random_embedding_matrix(
        vocab = get_all_unique_labels(features)
    )
    print('loaded label embedding')
    
    
    tokenizer = Tokenizer(word2id, pos2id, label2id)
    tokenized_features = tokenizer.tokenize_batch(features)
    
    embed()
    exit()
    
    
    
    # train_dataset = torch.utils.data.TensorDataset(torch.tensor(features), torch.tensor(labels))
    # dev_dataset = torch.utils.data.TensorDataset(torch.tensor(features), torch.tensor(labels))
    
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=32, shuffle=True)
    
    
    
    
    
    