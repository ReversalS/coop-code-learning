import json
import pickle

import pandas as pd
import os
from tqdm import tqdm

from multiprocessing import Pool


class Pipeline:
    def __init__(self, ratio, root):
        self.ratio = ratio
        self.root = root

    def preprocess(self, schema, pn=4, args_list):
        """
        preprocessing with multiple view choice and processes
        
        suppose we have three schema: A, B, and C
        and we will get: pid0-A, pid0-B, pid0-C, pid1-A, pid1-B, pid1-C, ...
        finally we merge them to get data-A, data-B, data-C
        """
        p = Pool(pn)
        p.map(schema, args_list)

        # merge all of the data so that we can split

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

    def tensorize(self, field_class, data_path):
        "tensorize raw data by field"
        field = field_class()
        data = load(data_path)
        field.build_vocab()

    # run for processing data to train
    def run(self):
        
        def _shema(list_of_files, output_prefix):
            # read files
            # preprocess by means of different schema
            # assert consistency (to make sure we actually get aligned views)
            # dump into files

        print("Start to preprocess dataset...")
        self.preprocess()


ppl = Pipeline('3:1:1', './data/')
ppl.run()
