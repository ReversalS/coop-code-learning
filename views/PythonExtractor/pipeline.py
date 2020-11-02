import json
import pickle

import pandas as pd
import os
from tqdm import tqdm


class Pipeline:
    def __init__(self, ratio, root):
        self.ratio = ratio
        self.root = root

        # the tuples of function names and its places
        # e.g It may like [['FUNCNAME1', 1, 35], ['FUNCNAME2', 3, 49],...]
        #     It means: in the first json tree, the 35th node's type == 'FunctionDef'
        #               in the third json tree, the 49th node's type == 'FunctionDef'
        self.sources = None

        # the json trees file
        # example can be found in ./data/cjson.txt
        # note that every line in this file is a json tree
        self.source_json_file = None

        # the Dict of path
        self.pc_dict = None
        self.pc_dict_num = 0

    # find all the node whose type == 'FunctionDef'
    # the results are pushed into self.sources
    def load_json_and_find_FunctionDef(self, filename):
        # DFS
        def traverse_in_json(node_idx):
            if js[node_idx]['type'] == 'FunctionDef':
                self.sources.append([js[node_idx]['value'], json_idx, node_idx])
            if 'children' in js[node_idx]:
                for child in js[node_idx]['children']:
                    traverse_in_json(child)

        path = self.root + filename
        self.source_json_file = path
        self.sources = []
        with open(path, 'r') as f:
            for json_idx, line in enumerate(tqdm(f.readlines())):
                js = json.loads(line)
                if len(js) == 0:
                    continue
                traverse_in_json(0)
        print(len(self.sources))

    # split data for training, developing and testing
    def split_data(self, cached=False):
        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        def _to_txt(filename, list_data):
            with open(filename, 'w') as f:
                for i in list_data:
                    f.write("%s\t%d\t%d\n" % (i[0], i[1], i[2]))

        train_file_path = self.root + 'train_.txt'
        dev_file_path = self.root + 'dev_.txt'
        test_file_path = self.root + 'test_.txt'

        if cached:
            if os.path.exists(train_file_path) and \
                    os.path.exists(dev_file_path) and \
                    os.path.exists(test_file_path):
                return

        data = self.sources
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
        _to_txt(train_file_path, train)
        _to_txt(dev_file_path, dev)
        _to_txt(test_file_path, test)

    # get pc
    def dump_path_context_dataset(self, max_path_length, max_path_width):
        "Use the json trees to parse context (of entire program)"

        from views.PythonExtractor.getpath import get_program_paths
        # words to ID by using self.pc_dict
        def compress_with_dict(path):
            words = path.split('-')
            compressed_pc = []
            for w in words:
                if w not in self.pc_dict:
                    self.pc_dict_num += 1
                    self.pc_dict[w] = self.pc_dict_num
                compressed_pc.append(self.pc_dict[w])
            return compressed_pc

        def extract_context(input_path, output_path):
            dataset = []

            # e.g It may like [['FUNCNAME1', 1, 35], ['FUNCNAME2', 3, 49],...]
            # note that they are sorted by the second para, which is the line ID of json tree
            tree_tuples = pd.read_table(input_path, header=None).values

            cur_idx = 0
            num_tuples = len(tree_tuples)
            with open(self.source_json_file, 'r') as ftree:
                # since the json trees file may be vary large, we don't want to read them all into memory
                # so, each time we read ONE line (or you can call it one tree) into memory,
                # and run `get_program_paths` for all the related tuples
                for line_idx, line in enumerate(tqdm(ftree.readlines())):
                    # if there are 3 functions in this tree, we will run it 3 times
                    while line_idx == tree_tuples[cur_idx][1]:
                        json_tree = json.loads(line)
                        path_contexts = get_program_paths(json_tree, tree_tuples[cur_idx][2], max_path_length,
                                                          max_path_width)
                        for path_context in path_contexts:
                            dataset.append(compress_with_dict(path_context))
                        cur_idx += 1
                        if cur_idx == num_tuples:
                            break
                    if cur_idx == num_tuples:
                        break
            with open(output_path, 'w') as f:
                for i in dataset:
                    f.write('-'.join([str(j) for j in i]) + '\n')

        self.pc_dict = {}
        self.pc_dict_num = 0
        extract_context(self.root + 'train_.txt', self.root + 'train_path_context.txt')
        extract_context(self.root + 'dev_.txt', self.root + 'dev_path_context.txt')
        extract_context(self.root + 'test_.txt', self.root + 'test_path_context.txt')
        with open(self.root + "dict.pkl", "wb") as f:
            pickle.dump(self.pc_dict, f)

    # run for processing data to train
    def run(self):
        print('load json dataset...')
        self.load_json_and_find_FunctionDef('cjson.txt')
        print('split data...')
        self.split_data(cached=True)
        print("Dump path contexts...")
        self.dump_path_context_dataset(max_path_length=10, max_path_width=2)
        # print("Filter path contexts...")
        # self.filter_path_context_dataset()

        # TODO list
        # print('train word embedding...')
        # self.dictionary_and_embedding(None, 128)
        # print('generate block sequences...')
        # self.generate_block_seqs(self.train_file_path, 'train')
        # self.generate_block_seqs(self.dev_file_path, 'dev')
        # self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline('3:1:1', './data/')
ppl.run()
