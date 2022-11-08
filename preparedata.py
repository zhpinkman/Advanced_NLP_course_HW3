from IPython import embed
from typing import List, Tuple, Dict
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import os


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

    def get_graph(self):
        G = nx.DiGraph()
        for token in self.tokens.values():
            G.add_node(token.token_id, word=token.word, pos_tag=token.pos)
        for token in self.tokens.values():
            if token.parent == 0:
                G.add_edge('root', token.token_id, label=token.label)
            else:
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
            return None
        return self.node_edges[index]

    def get_left_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return None
        return self.node_edges[index][0]

    def get_right_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return None
        return self.node_edges[index][-1]

    def get_left_most_child_left_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return None
        return self.get_left_most_child(
            self.node_edges[index][0].token_id
        )

    def get_right_most_child_right_most_child(self, index):
        if len(self.node_edges[index]) == 0:
            return None
        return self.get_right_most_child(
            self.node_edges[index][-1].token_id
        )


def read_sentences(file: str):
    with open(file) as f:
        lines = f.read().splitlines()
    sentences_tokens = []

    tokens = dict()
    for line in lines:
        if line == '':
            if len(tokens) > 0:
                sentences_tokens.append(tokens)
            tokens = dict()
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

    sentences_tokens = read_sentences('train.orig.conll')
    sentences = [Sentence(tokens) for tokens in sentences_tokens]
    sentences[0].draw_graph()
    embed()
    exit()
