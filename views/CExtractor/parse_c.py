import pycparser
import json as json


def parse_file(filename):
    tree = pycparser.parse_file(filename, use_cpp=False)

    json_tree = []

    def gen_identifier(identifier, node_type='identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos

    def traverse(node, node_type=None):
        def traverse_child(k, v):
            if v is None:
                children.append(gen_identifier("NULL", k))
            elif isinstance(v, list):
                children.append(traverse(v, k))
            elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) \
                    or isinstance(v, complex) or isinstance(v, str) or isinstance(v, set):
                children.append(gen_identifier(str(v), k))
            else:
                children.append(traverse(v))

        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        if node_type is None:
            json_node['type'] = type(node).__name__
        else:
            json_node['type'] = node_type

        if node is None:
            return pos

        children = []
        if isinstance(node, list):
            for l in node:
                traverse_child(json_node['type'] + '_list', l)
        else:
            for k, v in node.children():
                traverse_child(k, v)

        if len(children) != 0:
            json_node['children'] = children
        return pos

    traverse(tree)
    return json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False)


if __name__ == '__main__':
    filename = "tmp.c"
    print(parse_file(filename))
