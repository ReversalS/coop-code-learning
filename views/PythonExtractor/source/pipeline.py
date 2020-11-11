import json
import pickle
import ast
import os
from tqdm import tqdm
from multiprocessing import Pool


class FunctionField:
    def __init__(self, funcName, json_idx, node_idx, funcStr=None):
        # function name
        self.functionName = funcName

        # the ID of the json tree
        self.json_idx = json_idx

        # the node ID in the json tree
        # NOTE THAT node_idx<0 means this FunctionField is invalid
        # since there can be many functions in one file share the share function name,
        # we choose to filter them out by set their node_id=-1
        self.node_idx = node_idx

        # the source code of this function
        self.funcStr = funcStr


class Pipeline:
    def __init__(self, ratio, root):
        self.ratio = ratio
        self.root = root
        self.json_file = self.root + 'python100k_train.json'
        self.py150_source_file_names = self.root + 'python100k_train.txt'
        self.py150_sourc_dir = self.root

    # def preprocess(self, schema, pn=4, args_list):
    #     """
    #     preprocessing with multiple view choice and processes
    #
    #     suppose we have three schema: A, B, and C
    #     and we will get: pid0-A, pid0-B, pid0-C, pid1-A, pid1-B, pid1-C, ...
    #     finally we merge them to get data-A, data-B, data-C
    #     """
    #     p = Pool(pn)
    #     p.map(schema, args_list)

    # merge all of the data so that we can split

    # split data for training, developing and testing
    def split_data(self, cached=False):

        # it returns a List of FunctionField
        def generate_function_tuples():
            # DFS to find all the node whose type is 'FunctionDef' in json file
            # When we find two functions share the same name,
            # we mark their node_idx by -1 in order to filter them out
            def traverse_in_json(node_idx):
                if json_tree[node_idx]['type'] == 'FunctionDef':
                    if json_tree[node_idx]['value'] in function_dict:
                        function_dict[json_tree[node_idx]['value']].node_idx = -1
                    else:
                        function_dict[json_tree[node_idx]['value']] = \
                            FunctionField(json_tree[node_idx]['value'], json_idx, node_idx)
                if 'children' in json_tree[node_idx]:
                    for child in json_tree[node_idx]['children']:
                        traverse_in_json(child)
            
            import tokenize
            from io import BytesIO
            from srcseq.generate_data import my_tokenize
            # find all the node whose type is 'FunctionDef' in json file in ast
            def traverse_in_ast(tree):
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name in function_dict and function_dict[node.name].node_idx >= 0:
                            funcStr = ast.unparse(node)
                            funcStr = list(tokenize.tokenize(BytesIO(funcStr.encode('utf-8')).readline))
                            # funcStr = my_tokenize(funcStr)
                            # function_dict[node.name].funcStr = funcStr[funcStr.find('\n') + 1:]
                            function_dict[node.name].funcStr = funcStr

            def read_file_to_string(filename):
                f = open(filename, 'r')
                s = f.read()
                f.close()
                return s

            function_tuples = []
            with open(self.json_file, 'r') as f1, open(self.py150_source_file_names, 'r', encoding='UTF-8') as f2:
                for json_idx, json_tree_str in enumerate(tqdm(f1.readlines())):
                    function_dict = {}
                    json_tree = json.loads(json_tree_str)
                    source_code_file = self.py150_sourc_dir + f2.readline()[:-1]  # remove '\n'
                    if len(json_tree) == 0:
                        continue
                    traverse_in_json(0)
                    try:
                        tree = ast.parse(read_file_to_string(source_code_file))
                        traverse_in_ast(tree)
                        for functionItem in function_dict.values():
                            if functionItem.node_idx >= 0:
                                function_tuples.append(
                                    (functionItem.functionName,
                                    functionItem.json_idx,
                                    functionItem.node_idx,
                                    functionItem.funcStr))
                    except FileNotFoundError:
                        print("Early stopping of preprocessing")
                        break
            return function_tuples

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

        # def _shema(list_of_files, output_prefix):
        #     # read files
        #     # preprocess by means of different schema
        #     # assert consistency (to make sure we actually get aligned views)
        #     # dump into files
        #
        # print("Start to preprocess dataset...")
        # self.preprocess()

        self.split_data(cached=True)

ppl = Pipeline('3:1:1', './data/mnp/py150/')
ppl.run()
