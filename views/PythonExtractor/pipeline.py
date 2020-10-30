import json
import pickle

import pandas as pd
import os
from tqdm import tqdm


class Pipeline:
    def __init__(self, ratio, root):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.source_json_file = None
        self.pc_dict = None
        self.pc_dict_num = 0

    def load_json_and_find_FunctionDef(self, filename):
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
            # train_path = self.root + 'train/'
            # check_or_create(train_path)
            # self.train_file_path = train_path + 'train_.pkl'
            # dev_path = self.root + 'dev/'
            # check_or_create(dev_path)
            # self.dev_file_path = dev_path + 'dev_.pkl'
            # test_path = self.root + 'test/'
            # check_or_create(test_path)
            # self.test_file_path = test_path + 'test_.pkl'
            # return

        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)
        # random.shuffle(data) TODO: shuffle dataset by a fix random seed
        train = data[:train_split]
        dev = data[train_split:val_split]
        test = data[val_split:]

        train = sorted(train, key=lambda x: x[1])
        dev = sorted(dev, key=lambda x: x[1])
        test = sorted(test, key=lambda x: x[1])

        _to_txt(train_file_path, train)
        _to_txt(dev_file_path, dev)
        _to_txt(test_file_path, test)

    def dump_path_context_dataset(self, max_path_length, max_path_width):
        "Use the json trees to parse context (of entire program)"

        from views.PythonExtractor.getpath import get_program_paths
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
            tree_tuples = pd.read_table(input_path, header=None).values
            cur_idx = 0
            num_tuples = len(tree_tuples)
            with open(self.source_json_file, 'r') as ftree:
                for line_idx, line in enumerate(tqdm(ftree.readlines())):
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

    # def filter_path_context_dataset(self):
    #     "To somehow reduce total path count by removing useless(?) paths."
    #
    #     def filter_(input_path, output_path):
    #         with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    #             filtered = filter_paths(json.load(fin))
    #             json.dump(filtered, fout)
    #
    #     filter_(self.train_file_path + 'path_context.json', self.train_file_path + 'path_context_filtered.json')
    #     filter_(self.dev_file_path + 'path_context.json', self.dev_file_path + 'path_context_filtered.json')
    #     filter_(self.test_file_path + 'path_context.json', self.test_file_path + 'path_context_filtered.json')

    # run for processing data to train
    def run(self):
        print('load json dataset...')
        self.load_json_and_find_FunctionDef('cjson.txt')
        print('split data...')
        self.split_data(cached=True)
        print("Dump path contexts...")
        self.dump_path_context_dataset(max_path_length=10, max_path_width=2)
        print("Filter path contexts...")
        # self.filter_path_context_dataset()

        # print('train word embedding...')
        # self.dictionary_and_embedding(None, 128)
        # print('generate block sequences...')
        # self.generate_block_seqs(self.train_file_path, 'train')
        # self.generate_block_seqs(self.dev_file_path, 'dev')
        # self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline('3:1:1', './data/')
ppl.run()
