import argparse

parser = argparse.ArgumentParser(description='make translation tsv')

parser.add_argument('--source_path', type=str, default=None,
                    help='location of the data corpus')
parser.add_argument('--target_path', type=str, default = None,
                    help='location, of pretrained init')
parser.add_argument('--out_file', type=str, default = 'translation.tsv')


args = parser.parse_args()

with open(args.source_path, 'r') as source:
    with open(args.target_path, 'r') as tgt:
        with open(args.out_file, 'w') as out_file:

            for src_line, tgt_line in zip(source, tgt):

                print([src_line, tgt_line])
                src_line = src_line.strip('\n')
                tgt_line = tgt_line.strip('\n')
                out_file.write(src_line + '\t' + tgt_line + '\n')

