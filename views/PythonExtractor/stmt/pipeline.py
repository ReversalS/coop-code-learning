from __future__ import print_function, division, with_statement, unicode_literals
import json
import pickle
import ast
import os
from tqdm import tqdm
from multiprocessing import Pool
# import joblib
import itertools


class Pipeline:
    def __init__(self, ratio, root):
        self.ratio = ratio
        self.root = root
        self.json_file = self.root + 'python100k_train.json'
        self.py150_source_file_names = self.root + 'python100k_train.txt'
        self.py150_source_dir = self.root
        self.n_jobs = 4
        self.size = None

    def dictionary_and_embedding(self, size):
        train_file_path = self.root + 'py150json.txt'
        self.size = size
        if not os.path.exists(self.root + 'train/embedding'):
            os.mkdir(self.root + 'train/embedding')

        def get_sequences(node, ast, sequence):
            def get_token(node, lower=True):
                if 'value' in node.keys():
                    token = node['value']
                else:
                    token = node['type']
                if lower:
                    token = token.lower()
                return token

            sequence.append(get_token(node))
            if 'children' in node.keys():
                for child in node['children']:
                    get_sequences(ast[child], ast, sequence)
            if 'value' in node.keys() and node['value'] == 'compoundType':
                sequence.append('End')

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast[0], ast, sequence)
            return sequence

        corpus = []
        with open(train_file_path, 'r') as f:
            for line in f.readlines():
                json_tree = json.loads(line)
                corpus.append(trans_to_sequences(json_tree))
            #     str_corpus = [' '.join(c) for c in corpus]
            # trees['source'] = pd.Series(str_corpus)
            # trees.to_csv(self.root + 'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root + 'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self, data_path, dump_path):
        from gensim.models.word2vec import Word2Vec
        def get_token(node, lower=True):
            if 'value' in node.keys():
                token = node['value']
            else:
                token = node['type']
            if lower:
                token = token.lower()
            return token

        def get_blocks(node, ast, block_seq):
            if node['type'] in ['FunctionDef', 'If', 'For', 'While', 'DoWhile']:
                block_seq.append(node)

                if 'children' in node.keys():
                    for child in node['children']:
                        child = ast[child]
                        if 'value' in node.keys() and node['value'] == 'CompoundType' \
                                or node['type'] not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                            block_seq.append(child)
                        get_blocks(child, ast, block_seq)
            elif 'value' in node.keys() and node['value'] == 'CompoundType':
                block_seq.append(node)

                if 'children' in node.keys():
                    for child in node['children']:
                        child = ast[child]
                        if node['type'] not in ['If', 'For', 'While', 'DoWhile']:
                            block_seq.append(child)
                        get_blocks(child, ast, block_seq)

                block_seq.append({'type': 'End'})
            else:
                if 'children' in node.keys():
                    for child in node['children']:
                        child = ast[child]
                        get_blocks(child, ast, block_seq)

        def tree_to_index(node,ast):
            token = get_token(node)
            result = [vocab[token].index if token in vocab else max_token]
            if 'children' in node.keys():
                for child in node['children']:
                    result.append(tree_to_index(ast[child],ast))
            return result

        def trans2seq(r, ast):
            blocks = []
            get_blocks(r, ast, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b,ast)
                tree.append(btree)
            return tree

        word2vec = Word2Vec.load(self.root + 'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        train_file_path = self.root + 'py150json.txt'
        tree_tuples = pickle.load(open(data_path,'rb'))
        num_tuples = len(tree_tuples)
        all_blocks = []
        cur_idx = 0
        with open(train_file_path, 'r') as ftree:
            # since the json trees file may be vary large, we don't want to read them all into memory
            # so, each time we read ONE line (or you can call it one tree) into memory,
            # and run `get_program_paths` for all the related tuples
            for line_idx, line in enumerate(tqdm(ftree.readlines())):
                # if there are 3 functions in this tree, we will run it 3 times
                while line_idx == tree_tuples[cur_idx][1]:
                    json_tree = json.loads(line)
                    all_blocks.append(trans2seq(json_tree[tree_tuples[cur_idx][2]], json_tree))
                    cur_idx += 1
                    if cur_idx == num_tuples:
                        break
                if cur_idx == num_tuples:
                    break
        with open(dump_path, 'wb') as f:
            pickle.dump(all_blocks, f)

    # run for processing data to train
    def run(self):
        print('train word embedding...')
        self.dictionary_and_embedding(128)
        print('generate block sequences...')
        self.generate_block_seqs(self.root + '_train_.pkl', self.root + 'stmt_train.pkl')
        self.generate_block_seqs(self.root + '_dev_.pkl', self.root + 'stmt_dev.pkl')
        self.generate_block_seqs(self.root + '_test_.pkl', self.root + 'stmt_test.pkl')


ppl = Pipeline('3:1:1', '../data/')
ppl.run()
