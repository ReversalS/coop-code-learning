import pycparser
from pycparser import c_parser
import json as json


def parse_c_code(source_code, parser, filename=None):
    tree = parser.parse(source_code, filename)
    json_tree = []

    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__

        if len(node.attr_names) != 0:
            val = '#'.join([str(node.__getattribute__(i)) for i in node.attr_names if
                        ((node.__getattribute__(i) is not None) and (len(node.__getattribute__(i)) != 0))])
            if val != "":
                json_node['value'] = val

        if node is None:
            return pos

        children = []
        for k, v in node.children():
            children.append(traverse(v))

        if len(children) != 0:
            json_node['children'] = children
        return pos

    traverse(tree)
    return json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False)


def parse_file(filename):
    tree = pycparser.parse_file(filename, use_cpp=False)
    json_tree = []

    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__

        if len(node.attr_names) != 0:
            json_node['value'] = '#'.join([str(node.__getattribute__(i)) for i in node.attr_names if len(node.__getattribute__(i))!=0])

        if node is None:
            return pos

        children = []
        for k, v in node.children():
            children.append(traverse(v))

        if len(children) != 0:
            json_node['children'] = children
        return pos

    traverse(tree)
    return json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False)


if __name__ == '__main__':
    filename = "tmp.c"
    print(parse_file(filename))
