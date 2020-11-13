#!/usr/bin/env/python2

import ast
import token
import tokenize
from io import BytesIO
from srcseq.generate_data import Unparser, MyListFile


class FunctionField:
    def __init__(self, funcName, json_idx, node_idx, raw_token_seq=None):
        # function name
        self.functionName = funcName

        # the ID of the json tree
        self.json_idx = json_idx

        # the node ID in the json tree
        # NOTE THAT node_idx<0 means this FunctionField is invalid
        # since there can be many functions in one file share the share function name,
        # we choose to filter them out by set their node_id=-1
        self.node_idx = node_idx

        # the raw tokenized sequence of this function
        self.raw_token_seq = raw_token_seq


def process(json_idx, json_tree, ast_instance):
    "returns function_dict"

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

    # find all the node whose type is 'FunctionDef' in json file in ast
    def traverse_in_ast(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in function_dict and function_dict[node.name].node_idx >= 0:
                    # raw_token_seq = ast.unparse(node)
                    # raw_token_seq = list(tokenize.tokenize(BytesIO(raw_token_seq.encode('utf-8')).readline))
                    # function_dict[node.name].raw_token_seq = raw_token_seq

                    lst = MyListFile()
                    Unparser(node, lst)
                    function_dict[node.name].raw_token_seq = list(lst)
        
    function_dict = {}
    traverse_in_json(0)
    traverse_in_ast(ast_instance)
    return function_dict


def filter_tokens(func):
    """
    Post-process tokenized python function. The goal is to:
    1. remove function name, docstring, and get rid of anything 
    that provides extra information to the model at the stage of 
    preprocessing
    2. preserve only the required tokens, form real tokenized 
    sequence
    """

    ####### filter raw sequence preprocessed by tokenize ######
    # seq = ['<' + token.tok_name[tok.type] + '>'
    #  if (tok.string == '' or tok.string[0] == ' ') else tok.string 
    #  for tok in func]
    # for i, tok in enumerate(seq):
    #     if tok == 'def':
    #         del seq[i+1]
    #         break

    ####### filter raw sequence preprocessed by Unparser ######
    seq = ['<' + tok.type + '>'
     if (tok.text == '' or tok.text[0] == ' ') else tok.text
     for tok in func]
    for i, tok in enumerate(seq):
        if tok == 'def':
            del seq[i+1]
            break

    return seq
