

class Field:

    def __init__(self):
        self.fix_length = 
        self.tokenize = 'default'
        self.batch_first = True
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

    def build_vocab(self, data):
        pass

    def preprocess(self, x):
        "load a single example using this field"
        pass

    def process(self, batch):
        "process a list of examples to create a tensor"
        pass