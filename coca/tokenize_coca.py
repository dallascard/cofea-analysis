import os
import json
from glob import glob
from collections import Counter
from optparse import OptionParser

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COCA/',
                      help='Base directory: default=%default')
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model_type = options.model_type
    model_name_or_path = options.model
    tokenizer = options.tokenizer

    indir = os.path.join(basedir, 'clean')
    outdir = os.path.join(basedir, 'tokenized_' + model_name_or_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading model")
    if model_type == 'bert':
        tokenizer_class = BertTokenizer
    elif model_type == 'roberta':
        tokenizer_class = RobertaTokenizer
    else:
        raise ValueError("Model type not recognized")

    # Load pretrained model/tokenizer
    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    printed_empty = False

    token_counter = Counter()

    for infile in files:
        empty_docs = 0
        emtpy_after_tokenization = 0
        outlines = []
        basename = os.path.basename(infile)
        print(infile)
        with open(infile) as f:
            lines = f.readlines()
        print(len(lines))

        max_n_spans = 0
        max_n_pieces = 0
        gt_512 = 0
        for line in tqdm(lines):
            line = json.loads(line)
            line_id = line['id']
            # drop the header
            if 'text' not in line:
                print('No text in', line_id)
            else:
                text = line['text'].strip()

                if len(text) == 0:
                    empty_docs += 1
                    if not printed_empty:
                        print("Empty body in", line_id)
                        print(line)
                        printed_empty = True
                else:
                    # convert to tokens using BERT
                    raw_pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(text, add_special_tokens=False)]
                    max_n_pieces = max(max_n_pieces, len(raw_pieces))

                    rejoined_pieces = []
                    # rejoin into concatenated words
                    if len(raw_pieces) == 0:
                        emtpy_after_tokenization += 1
                        print("No tokens after tokenization in", line_id)
                        print(text)
                    else:
                        for p_i, piece in enumerate(raw_pieces):
                            if p_i == 0:
                                rejoined_pieces.append(piece)
                            elif piece.startswith('##'):
                                rejoined_pieces[-1] += piece
                            else:
                                rejoined_pieces.append(piece)
                        outline = {'id': line_id}
                        for field in ['year', 'genre']:
                            if field in line:
                                outline[field] = line[field]
                        outline['tokens'] = rejoined_pieces

                        token_counter.update(rejoined_pieces)

                        outlines.append(outline)

        print(basename, len(lines), empty_docs, emtpy_after_tokenization, len(outlines), len(lines) - empty_docs -len(outlines))
        outfile = os.path.join(outdir, basename)
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')

    outfile = os.path.join(outdir, 'token_counts.json')
    with open(outfile, 'w') as fo:
        json.dump(token_counter.most_common(), fo)

    print("Output written to", outdir)

if __name__ == '__main__':
    main()
