import torch
import pandas as pd
import numpy as np
import argparse
import pickle
import io
from IPython import embed
import os
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import preparedata
import joblib
torch.manual_seed(78)


class Tokenizer:
    def __init__(self, word2id, pos2id, label2id):
        self.word2id = word2id
        self.pos2id = pos2id
        self.label2id = label2id

    def tokenize_batch(self, features):
        all_features = []
        for sub_feature in features:
            sub_feature_tokenized = self.tokenize(sub_feature)
            all_features.append(sub_feature_tokenized)
        return np.array(all_features)

    def tokenize(self, features):
        results = []
        for word_feature in features[:18]:
            if word_feature in self.word2id:
                results.append(self.word2id[word_feature])
            else:
                results.append(self.word2id['<UNK>'])
        for pos_feature in features[18:36]:
            if pos_feature in self.pos2id:
                results.append(self.pos2id[pos_feature])
            else:
                results.append(self.pos2id['<UNK>'])

        for label_feature in features[36:]:
            if label_feature in self.label2id:
                results.append(self.label2id[label_feature])
            else:
                results.append(self.label2id['<UNK>'])
        return results


class DependencyParser(torch.nn.Module):
    def __init__(
        self,
        word_embedding,
        pos_embedding,
        label_embedding,
        n_features: int,
        hidden_dim: int,
        num_labels: int
    ) -> None:
        super().__init__()
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.label_embedding = label_embedding

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        self.f1 = torch.nn.Linear(48 * self.n_features, self.hidden_dim)
        self.f2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.f3 = torch.nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, x):
        word_features = x[:, 0:18]
        pos_features = x[:, 18:36]
        label_features = x[:, 36:]

        word_features = self.word_embedding(word_features)
        pos_features = self.pos_embedding(pos_features)
        label_features = self.label_embedding(label_features)

        word_features = word_features.reshape(-1, 18 * self.n_features)
        pos_features = pos_features.reshape(-1, 18 * self.n_features)
        label_features = label_features.reshape(-1, 12 * self.n_features)

        x = torch.cat((word_features, pos_features, label_features), dim=1)
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = self.f3(x)
        return x


def key_in_possible_actions(key, possible_actions):
    for possible_action in possible_actions:
        if key.startswith(possible_action):
            return True
    return False


def generate_output(configuration: preparedata.Configuration):
    results = []
    for token in configuration.sentence.tokens.values():
        if token.token_id == 0:
            continue
        output = [
            str(token.token_id),
            str(token.word),
            str(token.word),
            str(token.pos),
            str(token.pos),
            str('_'),
            str(token.predicted_parent),
            str(token.predicted_label),
            str('_'),
            str('_')
        ]
        results.append('\t'.join(output))
    results.append('')
    return results


def process_one_sentence(model, tokenizer, sentence, label_encoder, device):

    configuration = preparedata.Configuration(sentence=sentence)
    model = model.to(device)
    with torch.no_grad():
        while not configuration.is_finished():
            features = configuration.get_features()
            tokenized_features = tokenizer.tokenize(features)
            tokenized_features = torch.Tensor(
                tokenized_features).long().reshape(1, -1)
            tokenized_features = tokenized_features.to(device)
            output = model(tokenized_features)

            output_probabilities = torch.softmax(output, dim=1)
            labels_probablities = dict(
                zip(label_encoder.classes_, output_probabilities[0].cpu().numpy()))
            possible_actions, forbidden_actions = configuration.get_possible_actions()
            labels_probablities = {
                key: value for key, value in labels_probablities.items() if key_in_possible_actions(key, possible_actions) and (key not in forbidden_actions)}

            best_action = max(labels_probablities, key=labels_probablities.get)
            if best_action == 'shift':
                next_configuration_dict = configuration.shift()
                configuration = next_configuration_dict['new_configuration']
            elif best_action.startswith('left_arc'):
                next_configuration_dict = configuration.left_arc(
                    best_action[9:])
                configuration = next_configuration_dict['new_configuration']
            elif best_action.startswith('right_arc'):
                next_configuration_dict = configuration.right_arc(
                    best_action[10:])
                configuration = next_configuration_dict['new_configuration']
    output = generate_output(configuration)
    return output


def test_model(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    tokenizer = joblib.load('tokenizer.joblib')
    label_encoder = joblib.load('label_encoder.joblib')

    model = joblib.load(args.m)

    model = model.to(device)
    model.eval()

    sentences_tokens = preparedata.read_sentences(args.i)
    sentences = [preparedata.Sentence(tokens) for tokens in sentences_tokens]

    results = []
    for sentence in tqdm(sentences, leave=False):
        results.extend(process_one_sentence(
            model, tokenizer, sentence, label_encoder, device))

    results_string = '\n'.join(results)
    with open(args.o, 'w') as f:
        f.write(results_string)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help="input file")
    parser.add_argument('-o', type=str, help="output file")
    parser.add_argument('-m', type=str, help="model file")

    args = parser.parse_args()

    test_model(args)
