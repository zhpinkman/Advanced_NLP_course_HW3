import numpy as np


P_PREFIX = '<p>'
L_PREFIX = '<l>'
ROOT = '<root>'
NULL = '<null>'
UNK = '<unk>'


class Token:

    def __init__(self, token_id, word, pos, head, dep):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.head = head
        self.dep = dep
        self.predicted_head = -1
        self.predicted_dep = '<null>'
        self.lc, self.rc = [], []

    def reset_states(self):
        self.predicted_head = -1
        self.predicted_dep = '<null>'
        self.lc, self.rc = [], []


ROOT_TOKEN = Token(token_id=0, word=ROOT, pos=ROOT, head=-1, dep=ROOT)
NULL_TOKEN = Token(token_id=-1, word=NULL, pos=NULL, head=-1, dep=NULL)
UNK_TOKEN = Token(token_id=-1, word=UNK, pos=UNK, head=-1, dep=UNK)


class Sentence:

    def __init__(self, tokens):
        self.root = Token(token_id=0, word=ROOT, pos=ROOT, head=-1, dep=ROOT)
        self.tokens = None
        self.stack = None
        self.buffer = None
        self.arcs = None
        self.predicted_arcs = None

    def is_projective(self):
        """ determines if sentence is projective when ground truth given """
        pass


    def get_trans(self):  # this function is only used for the ground truth
        """ decide transition operation from [shift, left_arc, or right_arc] """
        pass


    def check_trans(self, potential_trans):
        """ checks if transition can legally be performed"""
        pass
    
    
    def update_state(self, curr_trans, predicted_dep=None):
        """ updates the sentence according to the given transition (may or may not assume legality, you implement) """
        pass



class FeatureGenerator:

    def __init__(self):
        pass

    def extract_features(self, sentence):
        """ returns the features for a sentence parse configuration """
        word_features = []
        pos_features = []
        dep_features = []

        return word_features, pos_features, dep_features
