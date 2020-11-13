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

    def preprocess(self):
        """
        preprocessing with multiple view choice and processes
    
        suppose we have three schema: A, B, and C
        and we will get: pid0-A, pid0-B, pid0-C, pid1-A, pid1-B, pid1-C, ...
        finally we merge them to get data-A, data-B, data-C

        NOTE: This part should be run in python2 environment!
        """
       
        from preprocess import process, filter_tokens

        def __collect(json_file_path, source_filenames_path, dump_path):
            function_tuples = []
            with open(json_file_path, 'r') as f1, open(source_filenames_path, 'r') as f2:
                for json_idx, json_tree_str in enumerate(tqdm(f1.readlines())):
                    if json_idx > 50:   # TEST
                        print('Force Early Stopping.')
                        break
                    json_tree = json.loads(json_tree_str)
                    source_code_file = self.py150_source_dir + f2.readline()[:-1]  # remove '\n'
                    if len(json_tree) == 0:
                        continue
                    try:
                        with open(source_code_file, 'r') as f:
                            tree = ast.parse(f.read())
                        function_dict = process(json_idx, json_tree, tree)
                        for functionItem in function_dict.values():
                            if functionItem.node_idx >= 0:
                                token_seq = filter_tokens(functionItem.raw_token_seq)
                                function_tuples.append(
                                    (functionItem.functionName,
                                    functionItem.json_idx,
                                    functionItem.node_idx,
                                    token_seq))
                    except IOError:  # no FileNotFoundError in py2
                        print("Early stopping of preprocessing")
                        break
            with open(dump_path, 'wb') as f:
                pickle.dump(function_tuples, f)
        
        __collect(
            self.root + 'python100k_train.json',
            self.root + 'python100k_train.txt',
            self.root + '__preprocessed_train.pkl'
        )
        __collect(
            self.root + 'python50k_eval.json',
            self.root + 'python50k_eval.txt',
            self.root + '__preprocessed_eval.pkl'
        )

    def split_data(self, cached=False):
        """
        split train data to (train, val)
        test (eval) data should not be split
        """

        def _to_pickle(filename, list_data):
            with open(filename, 'wb') as f:
                pickle.dump(list_data, f)

        train_file_path = self.root + '_train_.pkl'
        dev_file_path = self.root + '_dev_.pkl'
        test_file_path = self.root + '_test_.pkl'

        if cached:
            if os.path.exists(train_file_path) and \
                    os.path.exists(dev_file_path) and \
                    os.path.exists(test_file_path):
                return

        data = generate_function_tuples()
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)
        # random.shuffle(data) TODO: shuffle dataset by a fix random seed
        train = data[:train_split]
        dev = data[train_split:val_split]
        test = data[val_split:]

        # We sort the dataset tuples FOR THE USE IN dump_path_context_dataset::extract_context
        train = sorted(train, key=lambda x: x[1])
        dev = sorted(dev, key=lambda x: x[1])
        test = sorted(test, key=lambda x: x[1])

        # save them into txt file
        _to_pickle(train_file_path, train)
        _to_pickle(dev_file_path, dev)
        _to_pickle(test_file_path, test)

    # def tensorize(self, field_class, data_path):
    #     "tensorize raw data by field"
    #     field = field_class()
    #     data = load(data_path)
    #     field.build_vocab()

    # run for processing data to train
    def run(self):

        print("Start to preprocess dataset...")
        self.preprocess()

        # self.split_data(cached=True)

ppl = Pipeline('3:1:1', './data/mnp/py150/')
ppl.run()
