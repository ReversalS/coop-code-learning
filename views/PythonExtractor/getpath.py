import os
import argparse, logging
import json
from itertools import count, product
from collections import namedtuple

HalfPath = namedtuple('HalfPath', 'leaf_value, syntactic_half_path')
Path = namedtuple('Path', 'left_leaf_value, syntactic_path, right_leaf_value')
SyntacticPath = namedtuple('SyntacticPath', 'root_type, left_half, right_half')
SyntacticPath.__hash__ = lambda self: \
    hash(self.root_type) ^ hash(tuple(self.left_half)) ^ hash(tuple(self.right_half))

from tqdm import tqdm


class Method:
    target: str
    contexts: list

    def __init__(self, method_name, bag_of_paths):
        self.target = method_name
        self.contexts = bag_of_paths

    @staticmethod
    def convert(path: Path):
        return '{},{},{}'.format(
            path.left_leaf_value,
            hash(path.syntactic_path),
            path.right_leaf_value,
        )

    def to_string(self):
        target = self.target
        target = ''.join(c if c.isalpha() else ' ' for c in target)
        target = '|'.join(target.split())
        return target + ' ' + ' '.join([Method.convert(p) for p in self.contexts])

    @staticmethod
    def _print_path(path: Path):
        return path.left_leaf_value + '-' \
               + '-'.join(reversed(path.syntactic_path.left_half)) + '-' \
               + path.syntactic_path.root_type + '-' \
               + '-'.join(path.syntactic_path.right_half) + '-' \
               + path.right_leaf_value

    def print_path(self):
        return [Method._print_path(p) for p in self.contexts]


def get_program_paths(json_tree, func_idx, max_path_length=None, max_path_width=None):
    def traverse(i, level=0):
        node = json_tree[i]
        # ignore lookup
        if 'children' in node:
            bag_of_paths, root_leaf_paths_by_length = [], {}
            root_leaf_paths_by_length_by_ch = []
            for ch in node['children']:
                bag_of_paths_ch, root_leaf_paths_by_length_ch = traverse(ch, level)
                root_leaf_paths_by_length_by_ch.append(root_leaf_paths_by_length_ch)
                bag_of_paths.extend(bag_of_paths_ch)
                for length, paths in root_leaf_paths_by_length_ch.items():
                    if max_path_length is None or length < max_path_length:
                        root_leaf_paths_by_length \
                            .setdefault(length + 1, list()) \
                            .extend(
                            HalfPath(
                                leaf_value=p.leaf_value,
                                syntactic_half_path=[node['type']] + p.syntactic_half_path,
                            )
                            for p in paths)
            for i1, paths_by_length_ch1 in enumerate(root_leaf_paths_by_length_by_ch):
                for _, paths_by_length_ch2 in zip(
                        count(i1 + 1) if max_path_width is None else range(i1 + 1, i1 + max_path_width + 1),
                        root_leaf_paths_by_length_by_ch[i1 + 1:],
                ):
                    for len1, paths1 in paths_by_length_ch1.items():
                        if max_path_length is None or len1 < max_path_length:
                            for len2, paths2 in paths_by_length_ch2.items():
                                if max_path_length is None or len2 < max_path_length - len1 - 1:
                                    for p1, p2 in product(paths1, paths2):
                                        syntactic_path = SyntacticPath(
                                            root_type=node['type'],
                                            left_half=p1.syntactic_half_path,
                                            right_half=p2.syntactic_half_path,
                                        )
                                        bag_of_paths.append(Path(
                                            left_leaf_value=p1.leaf_value,
                                            syntactic_path=syntactic_path,  # todo: to string
                                            right_leaf_value=p2.leaf_value,
                                        ))
            return bag_of_paths, root_leaf_paths_by_length
        elif 'value' in node:
            ## is leaf and has token
            value = node['value']
            if node['type'] == "Str":
                value = ''.join(c for c in value if c.isalnum() or c in r'''_-''')
                if len(value) == 0:
                    value = "_"
            return [], {
                0: [
                    HalfPath(
                        leaf_value=value,
                        syntactic_half_path=[node['type']],
                    ),
                ],
            }
        else:
            return [], {}

    global_paths = []
    bag_of_paths, _ = traverse(func_idx)
    method = Method(json_tree[func_idx]['value'] if hasattr(json_tree[func_idx], 'value') else '', bag_of_paths)
    global_paths += method.print_path()
    return global_paths

