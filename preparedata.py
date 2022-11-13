from IPython import embed
from typing import List, Tuple, Dict
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import joblib


class Token:
    def __init__(self, token_id, word, pos, label, parent):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.parent = parent
        self.label = label

    def __repr__(self):
        inst = {
            'token_id': self.token_id,
            'word': self.word,
            'pos': self.pos,
            'parent': self.parent,
            'label': self.label
        }
        return str(inst)


class Sentence:
    def __init__(self, tokens: Dict[int, Token]):
        self.tokens = tokens
        self.__process()

    def get_child_parent_spans(self):
        spans = []
        for token in self.tokens.values():
            spans.append((token.token_id, token.parent))
        return spans

    def is_projective(self):
        spans = self.get_child_parent_spans()
        for i in range(len(spans)):
            for j in range(i+1, len(spans)):
                min_i, max_i = min(spans[i]), max(spans[i])
                min_j, max_j = min(spans[j]), max(spans[j])
                if min_i < min_j and max_i < max_j and max_i > min_j:
                    return False
                if min_i > min_j and max_i > max_j and min_i < max_j:
                    return False
        return True

    def get_graph(self):
        G = nx.DiGraph()
        for token in self.tokens.values():
            G.add_node(token.token_id, word=token.word, pos_tag=token.pos)

        for token in self.tokens.values():
            G.add_edge(token.parent, token.token_id, label=token.label)
        return G

    def draw_graph(self):
        fig = plt.figure(figsize=(20, 20))
        if os.path.exists('graph.png'):
            os.remove('graph.png')

        G = self.get_graph()
        pos = nx.nx_agraph.graphviz_layout(G)
        nx.draw(G, pos, labels={token_id: f"{token_id} - {token.word}" for token_id,
                token in self.tokens.items()}, with_labels=True)

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
        plt.savefig('graph.png')

    def __process(self):
        self.node_edges = defaultdict(list)
        for token in self.tokens.values():
            self.node_edges[token.parent].append(token)

    def __getitem__(self, index):
        return self.tokens[index]

    def get_children(self, index):
        if len(self.node_edges[index]) == 0:
            return []
        return self.node_edges[index]

    def get_left_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return 'None'
        node_to_return = self.node_edges[index][0]
        if node_to_return.token_id > index:
            return 'None'
        return node_to_return

    def get_second_left_most_child(self, index):
        if len(self.node_edges[index]) < 2:
            return 'None'
        node_to_return = self.node_edges[index][1]
        if node_to_return.token_id > index:
            return 'None'
        return node_to_return

    def get_right_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return 'None'
        node_to_return = self.node_edges[index][-1]
        if node_to_return.token_id < index:
            return 'None'
        return node_to_return

    def get_second_right_most_child(self, index):
        if len(self.node_edges[index]) < 2:
            return 'None'
        node_to_return = self.node_edges[index][-2]
        if node_to_return.token_id < index:
            return 'None'
        return node_to_return

    def get_left_most_child_left_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return 'None'
        first_child = self.node_edges[index][0]
        if first_child.token_id > index:
            return 'None'
        return self.get_left_most_child(
            first_child.token_id
        )

    def get_right_most_child_right_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return 'None'
        last_child = self.node_edges[index][-1]
        if last_child.token_id < index:
            return 'None'
        return self.get_right_most_child(
            last_child.token_id
        )


class Configuration:
    def __init__(self, sentence: Sentence) -> None:
        self.sentence = sentence
        self.stack = [0]
        self.buffer = [
            token_id for token_id in sentence.tokens.keys() if token_id != 0]
        self.all_processed = []

    def get_all_features(self):
        word_features, pos_features, label_features = self.get_features()
        assert len(word_features) == 18
        assert len(pos_features) == 18
        assert len(label_features) == 12

        all_features = word_features + pos_features + label_features

        return all_features

    def get_features(self):
        words = []
        poss = []
        labels = []

        for length in range(1, 4):
            if len(self.stack) >= length:
                words.append(self.sentence[self.stack[-length]].word)
                poss.append(self.sentence[self.stack[-length]].pos)
            else:
                words.append('None')
                poss.append('None')

        for length in range(1, 4):
            if len(self.buffer) >= length:
                words.append(self.sentence[self.buffer[length - 1]].word)
                poss.append(self.sentence[self.buffer[length - 1]].pos)
            else:
                words.append('None')
                poss.append('None')

        for length in range(1, 3):
            if len(self.stack) >= length:
                for func in [
                    self.sentence.get_left_most_child,
                    self.sentence.get_second_left_most_child,
                    self.sentence.get_right_most_child,
                    self.sentence.get_second_right_most_child
                ]:
                    fetched_node = func(self.stack[-length])
                    if fetched_node == 'None':
                        words.append('None')
                        poss.append('None')
                        labels.append('None')
                    else:
                        words.append(fetched_node.word)
                        poss.append(fetched_node.pos)
                        labels.append(fetched_node.label)

            else:
                for _ in range(4):
                    words.append('None')
                    poss.append('None')
                    labels.append('None')

        for length in range(1, 3):
            if len(self.stack) >= length:
                for func in [
                    self.sentence.get_left_most_child_left_most_child,
                    self.sentence.get_right_most_child_right_most_child
                ]:
                    fetched_node = func(self.stack[-length])
                    if fetched_node == 'None':
                        words.append('None')
                        poss.append('None')
                        labels.append('None')
                    else:
                        words.append(fetched_node.word)
                        poss.append(fetched_node.pos)
                        labels.append(fetched_node.label)
            else:
                for _ in range(2):
                    words.append('None')
                    poss.append('None')
                    labels.append('None')

        return words, poss, labels

    def __repr__(self) -> str:
        return f"Stack: {[self.sentence[token_id].word for token_id in self.stack]}, Buffer: {[self.sentence[token_id].word for token_id in self.buffer]}"

    def is_finished(self) -> bool:
        return len(self.buffer) == 0 and len(self.stack) == 1 and self.stack[0] == 0

    def __shift(self):
        past_configuration = copy.deepcopy(self)
        self.all_processed.append(self.buffer[0])
        self.stack.append(self.buffer.pop(0))
        return {
            'configuration': past_configuration,
            'action': 'shift',
            'new_configuration': self
        }

    def __left_arc(self):
        past_configuration = copy.deepcopy(self)
        label = self.sentence[self.stack[-2]].label
        self.stack.pop(-2)
        return {
            'configuration': past_configuration,
            'action': 'left_arc' + label,
            'new_configuration': self
        }

    def __right_arc(self):
        past_configuration = copy.deepcopy(self)
        label = self.sentence[self.stack[-1]].label
        self.stack.pop(-1)
        return {
            'configuration': past_configuration,
            'action': 'right_arc' + label,
            'new_configuration': self
        }

    def get_next_configuration(self):
        if self.is_finished():
            return None
        if len(self.stack) == 1:
            return self.__shift()

        if self.stack[-2] in [token.token_id for token in self.sentence.get_children(self.stack[-1])]:
            return self.__left_arc()

        if self.stack[-1] in [token.token_id for token in self.sentence.get_children(self.stack[-2])] and \
            all(
            [
                token_id in self.all_processed
                for token_id in [token.token_id for token in self.sentence.get_children(self.stack[-1])]
            ]
        ):
            return self.__right_arc()
        else:
            return self.__shift()


class Oracle:
    def __init__(self, sentences=List[Sentence]) -> None:
        self.sentences = sentences
        self.all_configurations = []
        self.all_labels = []
        print("Generating oracle...")
        count = 0
        for sentence in tqdm(self.sentences, leave=False):
            try:
                configurations, labels = self.get_configurations(sentence)
                self.all_configurations.append(configurations)
                self.all_labels.append(labels)
            except Exception as e:
                print(e)
                print(count)
                count += 1
        print('Total sentences with errors: ', count)
        print('Total sentences: ', len(self.sentences))

    def get_configurations(self, sentence):
        configurations = []
        actions = []

        configuration = Configuration(sentence)
        while not configuration.is_finished():
            configuration_dict = configuration.get_next_configuration()
            configuration = configuration_dict['new_configuration']
            configurations.append(configuration_dict['configuration'])
            actions.append(configuration_dict['action'])
        return configurations, actions


def read_sentences(file: str):
    with open(file) as f:
        lines = f.read().splitlines()
    sentences_tokens = []

    tokens = dict()
    tokens[0] = Token(0, 'ROOT', 'ROOT', 'ROOT', -1)
    for line in lines:
        if line == '':
            if len(tokens) > 0:
                sentences_tokens.append(tokens)
            tokens = dict()
            tokens[0] = Token(0, 'ROOT', 'ROOT', 'ROOT', -1)
            continue

        token = line.split('\t')
        # TODO: check what happens if you take the lemma of the word instead of the word itself
        tokens[int(token[0])] = Token(
            token_id=int(token[0]),
            word=token[1],
            pos=token[3],
            parent=int(token[6]),
            label=token[7],
        )

    return sentences_tokens


if __name__ == "__main__":
    mode = 'dev'
    if os.path.exists(f'{mode}.oracle.txt'):
        os.remove(f'{mode}.oracle.txt')

    sentences_tokens = read_sentences(f'{mode}.orig.conll')
    sentences = [Sentence(tokens) for tokens in sentences_tokens]
    print('number of sentences: ', len(sentences))
    sentences = [
        sentence for sentence in tqdm(sentences, leave=False) if sentence.is_projective()]
    print('number of projective sentences: ', len(sentences))

    parts = [(i, min(i + 5000, len(sentences)))
             for i in range(0, len(sentences), 5000)]
    print(parts)

    for part in tqdm(parts, leave=False):
        oracle = Oracle(
            sentences=sentences[part[0]:part[1]]
        )

        with open(f'{mode}.oracle.txt', 'a') as f:
            for sentence_configurations, sentence_labels in tqdm(zip(oracle.all_configurations, oracle.all_labels), leave=False):
                for configuration, label in zip(sentence_configurations, sentence_labels):
                    configuration_features = configuration.get_all_features()
                    output = '\t'.join([*configuration_features, label])
                    f.write(output + '\n')

    embed()
    exit()
