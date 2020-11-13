#!/usr/bin/env python2

import argparse
import ast
import json
import logging
import os
from collections import namedtuple
import tqdm

import sys
sys.path.append('.')
print(sys.path)

from srcseq.astunparser import Unparser, WriterBase

def file_tqdm(fobj):
    return tqdm(fobj, total=get_number_of_lines(fobj))


SrcASTToken = namedtuple("SrcASTToken", "text type lineno col_offset")
logging.basicConfig(level=logging.INFO)


class MyListFile(list, WriterBase):
    def write(self, text, type=None, node=None):
        text = text.strip()
        lineno = node and node.lineno
        col_offset = node and node.col_offset
        if len(text) > 0:
            # write `Str` as it is. `Num` will be kept as a string.
            text = eval(text) if type == "Str" else text
            self.append(SrcASTToken(text, type, lineno, col_offset))

    def flush(self):
        pass


def my_tokenize(code_str):
    t = ast.parse(code_str)
    lst = MyListFile()
    Unparser(t, lst)
    return lst


def main():
    parser = argparse.ArgumentParser(
        description="Generate datapoints from source code",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files_path", "-f", required=True,
        help="Filepath with the filenames to be parsed")
    parser.add_argument("--save", "-o", default="/tmp/dps.jsonl",
        help="Filepath with the output dps")
    parser.add_argument("--base_dir", "-b",
        help="Base dir to append for the fps."
        " If not given, use the dir of `--files_path`.")
    args = parser.parse_args()
    args.base_dir = args.base_dir or os.path.dirname(args.files_path)
    if os.path.exists(args.save):
        os.remove(args.save)

    num_dps = 0
    logging.info("Loading files from: {}".format(args.base_dir))
    with open(args.files_path, "r") as fin, open(args.save, "w") as fout:
        for i_line, line in enumerate(file_tqdm(fin)):
            rel_src_fp = line.strip()
            abs_src_fp = os.path.join(args.base_dir, rel_src_fp)
            try:
                values, types_, linenos, col_offsets = zip(*my_tokenize(open(abs_src_fp).read()))
                if len(values) > 1:
                    json.dump({
                        'rel_src_fp': rel_src_fp,
                        'values': values,
                        'types': types_,
                        'linenos': linenos,
                        'col_offsets': col_offsets,
                    }, fp=fout)
                    fout.write("\n")
                    num_dps += 1
                else:
                    # logging.info("In processing {}-th file `{}`: empty token list.".format(i_line, rel_src_fp))
                    pass
            except Exception as e:
                logging.warning("In processing {}-th file `{}`:\n\t{}".format(i_line, rel_src_fp, e))
                continue
    logging.info("Wrote {} datapoints to {}".format(num_dps, args.save))


if __name__ == "__main__":
    main()
