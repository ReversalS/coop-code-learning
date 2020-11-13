import pickle


class SourceCodeField:

    def __init__(self):
        # self.fix_length = 
        self.tokenize = 'default'
        self.batch_first = True
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.token2id = {}
        self.id2token = {}

    # def _tokenize(self, string):
        # pass

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            self.token2id = pickle.load(f)
            self.id2token = pickle.load(f)

    def build(self, corpus, vocab_size):
        def add_freq(d, key):
            if key not in d:
                d[key] = 1
            else:
                d[key] += 1
        
        freq = {}
        for line in corpus:
            for token in self._tokenize(line):
                try:
                    add_freq(freq, token)
                except ValueError:
                    print(repr(line))
                    assert 0
        freq = list(freq.items())
        freq.sort(key=lambda x: x[1], reverse=True)
        SKIP = 2
        for i, token in enumerate(freq[:vocab_size]):
            id = i + SKIP
            token = token[0]
            self.token2id[token] = id
            self.id2token[id] = token
    
    def dump(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.token2id, f)
            pickle.dump(self.id2token, f)

    def preprocess(self, x):
        "load a single example using this field"
        pass

    def process(self, dataset, unk_id=1):
        "process a list of examples to create a tensor"
        assert self.token2id != {}
        assert self.batch_first
        tensor_dataset = []
        for line in corpus:
            data = []
            for token in self._tokenize(line):
                if token in self.token2id:
                    tid = self.token2id[tid]
                else:
                    tid = unk_id
                data.append(tid)
            tensor_dataset.append(data)
        return tensor_dataset