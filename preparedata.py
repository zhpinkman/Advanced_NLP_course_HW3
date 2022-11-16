from IPython import embed
from typing import List, Tuple, Dict
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import argparse
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
        self.predicted_label = None
        self.predicted_parent = None
        self.left_children = []
        self.right_children = []

    def get_left_most_child(self):
        if len(self.left_children) == 0:
            return None
        else:
            return self.left_children[0]

    def get_right_most_child(self):
        if len(self.right_children) == 0:
            return None
        else:
            return self.right_children[-1]

    def get_second_left_most_child(self):
        if len(self.left_children) < 2:
            return None
        else:
            return self.left_children[1]

    def get_second_right_most_child(self):
        if len(self.right_children) < 2:
            return None
        else:
            return self.right_children[-2]

    def __repr__(self):
        inst = {
            'token_id': self.token_id,
            'word': self.word,
            'pos': self.pos,
            'parent': self.parent,
            'label': self.label,
            'predicted_label': self.predicted_label,
            'predicted_parent': self.predicted_parent,
            'left_children': self.left_children,
            'right_children': self.right_children
        }
        return str(inst)


class Sentence:
    def __init__(self, tokens: Dict[int, Token]):
        self.tokens = tokens

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
            for child in token.left_children:
                G.add_edge(token.token_id, child.token_id,
                           label=child.predicted_label)
            for child in token.right_children:
                G.add_edge(token.token_id, child.token_id,
                           label=child.predicted_label)

        # for token in self.tokens.values():
        #     if token.parent != -1:
        #         G.add_edge(token.parent, token.token_id, label=token.label)
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

    def process(self):
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
        return self.tokens[index].get_left_most_child()

    def get_second_left_most_child(self, index):
        return self.tokens[index].get_second_left_most_child()

    def get_right_most_child(self, index):
        return self.tokens[index].get_right_most_child()

    def get_second_right_most_child(self, index):
        return self.tokens[index].get_second_right_most_child()

    def get_left_most_child_left_most_child(self, index):
        left_most_child = self.tokens[index].get_left_most_child()
        if left_most_child is None:
            return None
        return left_most_child.get_left_most_child()

    def get_right_most_child_right_most_child(self, index):
        right_most_child = self.tokens[index].get_right_most_child()
        if right_most_child is None:
            return None
        return right_most_child.get_right_most_child()


class Configuration:
    def __init__(self, sentence: Sentence) -> None:
        self.sentence = sentence
        self.stack = [0]
        self.buffer = [
            token_id for token_id in sentence.tokens.keys() if token_id != 0]
        self.all_processed = []

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
                    if fetched_node is None:
                        words.append('None')
                        poss.append('None')
                        labels.append('None')
                    else:
                        words.append(fetched_node.word)
                        poss.append(fetched_node.pos)
                        labels.append(fetched_node.predicted_label)

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
                    if fetched_node is None:
                        words.append('None')
                        poss.append('None')
                        labels.append('None')
                    else:
                        words.append(fetched_node.word)
                        poss.append(fetched_node.pos)
                        labels.append(fetched_node.predicted_label)
            else:
                for _ in range(2):
                    words.append('None')
                    poss.append('None')
                    labels.append('None')

        assert len(words) == 18
        assert len(poss) == 18
        assert len(labels) == 12
        final_results = words + poss + labels
        return final_results

    def __repr__(self) -> str:
        return f"Stack: {[self.sentence[token_id].word for token_id in self.stack]}, Buffer: {[self.sentence[token_id].word for token_id in self.buffer]}"

    def is_finished(self) -> bool:
        return len(self.buffer) == 0 and len(self.stack) == 1 and self.stack[0] == 0

    def shift(self):
        past_configuration = copy.deepcopy(self)
        self.all_processed.append(self.buffer[0])
        self.stack.append(self.buffer.pop(0))
        return {
            'configuration': past_configuration,
            'action': 'shift',
            'new_configuration': self
        }

    def left_arc(self, label):
        past_configuration = copy.deepcopy(self)
        self.sentence[self.stack[-1]
                      ].left_children.append(self.sentence[self.stack[-2]])
        self.sentence[self.stack[-2]].predicted_label = label
        self.sentence[self.stack[-2]].predicted_parent = self.stack[-1]
        self.stack.pop(-2)
        return {
            'configuration': past_configuration,
            'action': 'left_arc' + '_' + label,
            'new_configuration': self
        }

    def right_arc(self, label):
        past_configuration = copy.deepcopy(self)
        self.sentence[self.stack[-2]
                      ].right_children.append(self.sentence[self.stack[-1]])
        self.sentence[self.stack[-1]].predicted_label = label
        self.sentence[self.stack[-1]].predicted_parent = self.stack[-2]
        self.stack.pop(-1)
        return {
            'configuration': past_configuration,
            'action': 'right_arc' + '_' + label,
            'new_configuration': self
        }

    def get_possible_actions(self):
        actions = []
        forbidden_actions = []
        if len(self.buffer) > 0:
            actions.append('shift')
        if len(self.stack) > 1:
            if len(self.stack) != 2:
                actions.append('left_arc')
        if len(self.stack) > 1:
            actions.append('right_arc')

        forbidden_actions.append('left_arc_root')
        if len(self.stack) != 1 or len(self.buffer) != 0:
            forbidden_actions.append('right_arc_root')
        return actions, forbidden_actions

    def get_next_configuration(self):
        if self.is_finished():
            return None
        if len(self.stack) == 1:
            return self.shift()

        if self.stack[-2] in [token.token_id for token in self.sentence.get_children(self.stack[-1])]:
            return self.left_arc(self.sentence[self.stack[-2]].label)

        if self.stack[-1] in [token.token_id for token in self.sentence.get_children(self.stack[-2])] and \
            all(
            [
                token_id in self.all_processed
                for token_id in [token.token_id for token in self.sentence.get_children(self.stack[-1])]
            ]
        ):
            return self.right_arc(self.sentence[self.stack[-1]].label)
        else:
            return self.shift()


class Oracle:
    def __init__(self, sentences: List[Sentence], mode: str) -> None:
        self.sentences = sentences
        print("Generating oracle...")
        count = 0
        for sentence in tqdm(self.sentences, leave=False):
            try:
                sentence_configurations, sentence_labels = self.get_configurations(
                    sentence)
                with open(f'{mode}.converted', 'a') as f:
                    for configuration, label in zip(sentence_configurations, sentence_labels):
                        configuration_features = configuration.get_features()
                        output = '\t'.join([*configuration_features, label])
                        f.write(output + '\n')
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'dev'])
    args = parser.parse_args()

    mode = args.mode

    if os.path.exists(f'{mode}.converted'):
        os.remove(f'{mode}.converted')

    sentences_tokens = read_sentences(f'{mode}.orig.conll')
    sentences = [Sentence(tokens) for tokens in sentences_tokens]
    for sentence in sentences:
        sentence.process()

    print('number of sentences: ', len(sentences))
    sentences = [
        sentence for sentence in tqdm(sentences, leave=False) if sentence.is_projective()]
    print('number of projective sentences: ', len(sentences))

    oracle = Oracle(
        sentences=sentences,
        mode=mode
    )

    embed()
    exit()
