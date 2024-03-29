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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


def get_all_unique_target_labels(features):
    file_name = 'unique_target_labels.joblib'
    if os.path.exists(file_name):
        print('Loading unique target labels from file')
        unique_target_labels = joblib.load(file_name)
        return unique_target_labels
    prefixes = ["right_arc", "left_arc"]
    all_labels_without_prefix = set()
    for label in features:
        for prefix in prefixes:
            if label.startswith(prefix):
                all_labels_without_prefix.add(label[len(prefix):])

    final_results = []
    final_results.append("shift")
    for label in all_labels_without_prefix:
        for prefix in prefixes:
            final_results.append(prefix + label)
    joblib.dump(final_results, file_name)
    return final_results


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


def create_random_embedding_matrix(vocab, n_features):
    word2id = {word: index for index, word in enumerate(vocab)}
    word2id['<UNK>'] = len(word2id)

    vocab_size = len(word2id)
    vector_size = n_features

    embedding_matrix = np.random.rand(vocab_size, vector_size)
    embedding = torch.nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=vector_size)
    embedding.weight = torch.nn.Parameter(
        torch.tensor(embedding_matrix, dtype=torch.float32))
    return embedding, word2id


def create_word_embedding_matrix(vocab, n_features):
    glove = pd.read_csv(f'glove.6B.{n_features}d.txt', sep=" ",
                        quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    embedding_matrix = np.zeros((len(vocab), n_features))

    word2id = dict()

    for index, word in enumerate(vocab):
        if word in glove_embedding:
            embedding_matrix[index] = glove_embedding[word]
        word2id[word] = index

    word2id['<UNK>'] = len(word2id)
    unk_matrix = np.zeros((1, n_features))

    embedding_matrix = np.concatenate((embedding_matrix, unk_matrix), axis=0)

    vocab_size = embedding_matrix.shape[0]
    vector_size = embedding_matrix.shape[1]

    embedding = torch.nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=vector_size)
    embedding.weight = torch.nn.Parameter(
        torch.tensor(embedding_matrix, dtype=torch.float32))
    return embedding, word2id


class DependencyParser(torch.nn.Module):
    def __init__(
        self,
        word_embedding,
        pos_embedding,
        label_embedding,
        n_features: int,
        hidden_dim: int,
        num_labels: int,
        activation_function: str,
        use_dropout: bool
    ) -> None:
        super().__init__()
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.label_embedding = label_embedding

        if activation_function == "relu":
            self.activation_function = torch.relu
        elif activation_function == "tanh":
            self.activation_function = torch.tanh

        if use_dropout:
            self.dropout = torch.nn.Dropout(p=0.5)
        else:
            self.dropout = torch.nn.Identity()

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

        x = self.dropout(x)

        x = self.activation_function(self.f1(x))
        x = self.activation_function(self.f2(x))
        x = self.f3(x)
        return x


def evaluate(model, loader, device):
    model.eval()
    all_predictions = []
    all_actual_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.tolist())
            all_actual_labels.extend(y.tolist())
    return {
        'accuracy': accuracy_score(all_actual_labels, all_predictions),
        'f1': f1_score(all_actual_labels, all_predictions, average='weighted'),
        'precision': precision_score(all_actual_labels, all_predictions, average='weighted'),
        'recall': recall_score(all_actual_labels, all_predictions, average='weighted')
    }


def compute_loss(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            total_samples += x.shape[0]

    return total_loss / total_samples


def train(model, criterion, loader, dev_loader, optimzer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)

        optimzer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimzer.step()

        total_loss += loss.item()
        total_samples += x.shape[0]

        if total_samples % 100000 == 0:
            print(f'Loss: {total_loss / total_samples}')
            dev_loss = compute_loss(model, criterion, dev_loader, device)
            print(f'Dev Loss: {dev_loss}')
    return total_loss / len(loader)


def train_model(args):
    train_features, train_labels = get_features_labels(args.train)
    dev_features, dev_labels = get_features_labels(args.dev)

    if args.data_ratio < 1:
        random_indices = np.random.choice(
            np.arange(len(train_features)),
            size=int(args.data_ratio * len(train_features)),
            replace=False
        )

        train_features = np.array([train_features[i] for i in random_indices])
        train_labels = np.array([train_labels[i] for i in random_indices])

    if args.random_word_embedding == 'True':
        word_embedding, word2id = create_random_embedding_matrix(
            vocab=get_all_unique_words(train_features),
            n_features=args.n_features
        )
    else:
        word_embedding, word2id = create_word_embedding_matrix(
            vocab=get_all_unique_words(train_features),
            n_features=args.n_features
        )
    print('loaded word embedding')
    pos_embedding, pos2id = create_random_embedding_matrix(
        vocab=get_all_unique_poss(train_features),
        n_features=args.n_features
    )
    print('loaded pos embedding')
    label_embedding, label2id = create_random_embedding_matrix(
        vocab=get_all_unique_labels(train_features),
        n_features=args.n_features
    )
    print('loaded label embedding')

    label_encoder = LabelEncoder()
    all_unique_train_labels = get_all_unique_target_labels(train_labels)
    label_encoder.fit(all_unique_train_labels)

    tokenized_train_labels = label_encoder.transform(train_labels)
    tokenized_dev_labels = label_encoder.transform(dev_labels)

    tokenizer = Tokenizer(word2id, pos2id, label2id)
    tokenized_train_features = tokenizer.tokenize_batch(train_features)
    tokenized_dev_features = tokenizer.tokenize_batch(dev_features)

    joblib.dump(tokenizer, 'tokenizer.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(tokenized_train_features), torch.tensor(tokenized_train_labels))

    dev_dataset = torch.utils.data.TensorDataset(
        torch.tensor(tokenized_dev_features), torch.tensor(tokenized_dev_labels))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=16, shuffle=False
    )

    model = DependencyParser(
        word_embedding=word_embedding,
        pos_embedding=pos_embedding,
        label_embedding=label_embedding,
        n_features=args.n_features,
        hidden_dim=args.hidden_dim,
        num_labels=len(label_encoder.classes_),
        activation_function=args.activation_function,
        use_dropout=args.use_dropout
    )
    model = model.to(device)

    # class_weights = compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.unique(tokenized_labels),
    #     y=tokenized_labels
    # )

    criterion = torch.nn.CrossEntropyLoss()

    # weight=torch.tensor(class_weights).float()

    criterion = criterion.to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.08, weight_decay=0.0001
        )
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=0.08, weight_decay=0.0001
        )
    else:
        raise NotImplementedError()

    print('training')

    best_dev_f1 = -np.inf

    for epoch in range(args.epochs):
        train_loss = train(model, criterion, train_dataloader,
                           dev_dataloader, optimizer, device)
        print(f'epoch: {epoch}, train_loss: {train_loss}')
        train_metrics = evaluate(model, train_dataloader, device)
        print(train_metrics)
        eval_metrics = evaluate(model, dev_dataloader, device)
        print(eval_metrics)
        if eval_metrics['f1'] > best_dev_f1:
            best_dev_f1 = eval_metrics['f1']
            model = model.to('cpu')
            joblib.dump(model, args.m)
            model = model.to(device)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str,
                        default="train.converted", help="path to train file")
    parser.add_argument("--dev", type=str,
                        default="dev.converted", help="path to dev file")
    parser.add_argument('-m', type=str, required=True, help='model name')
    parser.add_argument('--random_word_embedding',
                        action='store_true', help='use random word embedding')
    parser.add_argument('--optimizer', type=str,
                        default='adagrad', help='optimizer to use')
    parser.add_argument('--data_ratio', type=float,
                        default=1.0, help='ratio of data to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='hidden dimension of the fully connected layer')
    parser.add_argument('--n_features', type=int, default=50, choices=[
                        50, 100, 200, 300], help='number of features in the embedding layer')
    parser.add_argument('--activation_function', type=str,
                        default="relu", help='activation function to use')
    parser.add_argument('--use_dropout', action='store_true',
                        default=True, help='use dropout')

    args = parser.parse_args()
    print(args)

    train_model(args)
