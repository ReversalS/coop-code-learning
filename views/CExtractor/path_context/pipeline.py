# temporary pipeline for path-context

import pandas as pd
import os
import json
import pickle
from tqdm import tqdm
from pycparser import c_parser
from fields import PathContextField
from parse_c import parse_c_code
from getpath import get_program_paths
from path_filter import filter_paths

class Pipeline:
    "Assume dataset already built by ASTNN preprocess"
    def __init__(self, root):
        self.root = root
        self.train_file_path = root + 'train/'
        self.dev_file_path = root + 'dev/'
        self.test_file_path = root + 'test/'
        self.raw_dir = root + 'classification/' 
    
    def dump_json_trees(self):
        "Dump json format asts in one text file"
        parser = c_parser.CParser()

        def parse_(dataset, parser, output_path):
            json_tree_dataset = []
            problem_count = 0
            for cid, pn in tqdm(zip(dataset['class_id'], dataset['program_name'])):
                try:
                    with open(self.raw_dir + str(cid) + '/' + pn, 'r') as f:
                        json_tree_str = parse_c_code(f.read(), parser)
                        json_tree_dataset.append(json_tree_str)
                except TypeError:
                    # print(f'>>>> class_id: {cid}, program name: {pn}')
                    problem_count += 1
            print(f'total problems: {problem_count}')
            with open(output_path, 'w') as f:
                f.write('\n'.join(json_tree_dataset))

        train = pd.read_pickle(self.train_file_path + 'train_.pkl')
        parse_(train, parser, self.train_file_path + 'trees.json')
        dev = pd.read_pickle(self.dev_file_path + 'dev_.pkl')
        parse_(dev, parser, self.dev_file_path + 'trees.json')
        test = pd.read_pickle(self.test_file_path + 'test_.pkl')
        parse_(test, parser, self.test_file_path + 'trees.json')

    def dump_path_context_dataset(self, max_path_length, max_path_width):
        "Use the json trees to parse context (of entire program)"
        def extract_context(input_path, output_path):
            dataset = []
            with open(input_path, 'r') as f:
                for line in tqdm(f.readlines()):
                    json_tree = json.loads(line)
                    path_contexts = get_program_paths(json_tree, max_path_length, max_path_width)
                    dataset.append(path_contexts)
            with open(output_path, 'w') as f:
                json.dump(dataset, f)
        extract_context(self.train_file_path + 'trees.json', self.train_file_path + 'path_context.json')
        extract_context(self.dev_file_path + 'trees.json', self.dev_file_path + 'path_context.json')
        extract_context(self.test_file_path + 'trees.json', self.test_file_path + 'path_context.json')

    def filter_path_context_dataset(self):
        "To somehow reduce total path count by removing useless(?) paths."
        def filter_(input_path, output_path):
            with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
                filtered = filter_paths(json.load(fin))
                json.dump(filtered, fout)
        filter_(self.train_file_path + 'path_context.json', self.train_file_path + 'path_context_filtered.json')
        filter_(self.dev_file_path + 'path_context.json', self.dev_file_path + 'path_context_filtered.json')
        filter_(self.test_file_path + 'path_context.json', self.test_file_path + 'path_context_filtered.json')

    def build_vocab(self, token_vocab_size, path_vocab_size):
        field = PathContextField()
        with open(self.train_file_path + 'path_context_filtered.json', 'r') as f:
            corpus = json.load(f)
            field.build(corpus, token_vocab_size, path_vocab_size)
        field.dump(self.train_file_path + 'path_context_field.pkl')
    
    def dump_tensor_dataset(self):
        # right input format and process with field
        field = PathContextField()
        field.load(self.train_file_path + 'path_context_field.pkl')
        with open(self.train_file_path + 'data.pkl', 'wb') as fout, \
            open(self.train_file_path + 'path_context_filtered.json', 'r') as fin:
            dataset = field.process(json.load(fin))
            pickle.dump(dataset, fout)
        with open(self.dev_file_path + 'data.pkl', 'wb') as fout, \
            open(self.dev_file_path + 'path_context_filtered.json', 'r') as fin:
            dataset = field.process(json.load(fin))
            pickle.dump(dataset, fout)
        with open(self.test_file_path + 'data.pkl', 'wb') as fout, \
            open(self.test_file_path + 'path_context_filtered.json', 'r') as fin:
            dataset = field.process(json.load(fin))
            pickle.dump(dataset, fout)
    
    def dump_tensor_labels(self):
        train = pd.read_pickle(self.train_file_path + 'train_.pkl')
        pickle.dump(train['class_id'].to_numpy(), open(self.train_file_path + 'labels.pkl', 'wb'))
        dev = pd.read_pickle(self.dev_file_path + 'dev_.pkl')
        pickle.dump(dev['class_id'].to_numpy(), open(self.dev_file_path + 'labels.pkl', 'wb'))
        test = pd.read_pickle(self.test_file_path + 'test_.pkl')
        pickle.dump(test['class_id'].to_numpy(), open(self.test_file_path + 'labels.pkl', 'wb'))

    def run(self):
        print("Dump json trees...")
        # self.dump_json_trees()
        print("Dump path contexts...")
        # self.dump_path_context_dataset(max_path_length=10, max_path_width=2)
        print("Filter path contexts...")
        # self.filter_path_context_dataset()
        print("Dump field of path contexts...")
        # Total distinct tokens: 31413; total distinct paths: 522049
        # self.build_vocab(token_vocab_size=30000, path_vocab_size=50000)
        print("Tensorize datasets...")
        # self.dump_tensor_dataset()
        self.dump_tensor_labels()


ppl = Pipeline('data/poj_cls/')
ppl.run()