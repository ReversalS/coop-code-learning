import os
import pickle
from tqdm import tqdm


class PathContextField:
    "basic field (consider "

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.path2id = {}
        self.id2path = {}
        # self.target2id = {}
        # self.id2target

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            self.token2id = pickle.load(f)
            self.id2token = pickle.load(f)
            self.path2id = pickle.load(f)
            self.id2path = pickle.load(f)

    def build(self, corpus, token_vocab, path_vocab):
        """
        TODO: output some statistics for better dataset understanding
        1. total vocab
        2. max length, average length

        
        """
        token_freq = {}
        path_freq = {}
        
        def add_freq(d, key):
            if key not in d:
                d[key] = 1
            else:
                d[key] += 1

        for path_contexts in tqdm(corpus):
            for line in path_contexts:
                try:
                    left_token, path, right_token = line.split(',')[:3]
                    add_freq(token_freq, left_token)
                    add_freq(token_freq, right_token)
                    add_freq(path_freq, path)
                except ValueError:
                    print(line)
                    assert 0
        
        print(f'Total distinct tokens: {len(token_freq)}; total distinct paths: {len(path_freq)}')

        token_freq = list(token_freq.items())
        token_freq.sort(key=lambda x: x[1], reverse=True)
        path_freq = list(path_freq.items())
        path_freq.sort(key=lambda x: x[1], reverse=True)

        SKIP = 2    # 0 -> pad, 1 -> <unk>

        for i, token in enumerate(token_freq[:token_vocab]):
            id = i + SKIP
            token = token[0]
            self.token2id[token] = id
            self.id2token[id] = token
        for i, path in enumerate(path_freq[:path_vocab]):
            id = i + SKIP
            path = path[0]
            self.path2id[path] = id
            self.id2path[id] = path

    def dump(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.token2id, f)
            pickle.dump(self.id2token, f)
            pickle.dump(self.path2id, f)
            pickle.dump(self.id2path, f)

    def process(self, dataset, unk_id=1):
        assert self.token2id != {}
        tensor_dataset = []
        for path_contexts in tqdm(dataset):
            data = []
            for line in path_contexts:
                left_token, path, right_token = line.split(',')[:3]
                if left_token in self.token2id:
                    left_id = self.token2id[left_token]
                else:
                    left_id = unk_id
                if path in self.path2id:
                    path_id = self.path2id[path]
                else:
                    path_id = unk_id
                if right_token in self.token2id:
                    right_id = self.token2id[right_token]
                else:
                    right_id = unk_id
                tensor = [left_id, path_id, right_id]
                data.append(tensor)
            tensor_dataset.append(data)
        return tensor_dataset
