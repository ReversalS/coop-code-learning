import json
import pickle
import ast
import os
from tqdm import tqdm
from multiprocessing import Pool


class Pipeline:
    def __init__(self, ratio, json_path, root_path):
        self.ratio = ratio
        self.root = root_path
        self.json_path = self.json_path
        self.py150_source_file_names = self.root + 'python100k_train.txt'
        self.py150_source_dir = self.root + 'py150/'

    def preprocess(self, schema, pn=4, args_list):
        """
        preprocessing with multiple view choice and processes
    
        suppose we have three schema: A, B, and C
        and we will get: pid0-A, pid0-B, pid0-C, pid1-A, pid1-B, pid1-C, ...
        finally we merge them to get data-A, data-B, data-C
        """
        import .source
        import .path_context

        def process(json_tree, source_code):
            "process one tree/sc"
            tree = ast.parse(source_code)
            func_tuples = source.process(json_tree, tree)
            # path_contexts = path_context.process(json_tree)
                        
        # p = Pool(pn)
        # p.map(schema, args_list)

    # merge all of the data so that we can split

    # split data for training, developing and testing
    def split_data(self, data, cached=False):

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

        # def _schema(list_of_files, output_prefix):
        #     # read files
        #     # preprocess by means of different schema
        #     # assert consistency (to make sure we actually get aligned views)
        #     # dump into files
        #
        # print("Start to preprocess dataset...")
        # self.preprocess()

        self.split_data(cached=True)

ppl = Pipeline('3:1:1', './data/')
ppl.run()
