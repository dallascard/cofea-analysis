import os
import re
import json
from optparse import OptionParser

import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer

from common.alt_spellings import get_replacements
from common.misc import get_model_name


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/constitution/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--output-subdir', type=str, default='mlm_legal_vs_popular',
                      help='Output subdirectory: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    output_subdir = options.output_subdir
    alt_spellings = True
    seed = options.seed    
    np.random.seed(seed)

    model_name = get_model_name(model)

    print("Loading model")
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model)

    # get the tokenized forms of mispelling replacements
    replacements = get_replacements(tokenized=True)

    suffix = ''

    indir = os.path.join(basedir, 'tokenized_' + model_name)
    print("Using data from", indir)
    outdir = os.path.join(basedir, output_subdir + suffix)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    infile = os.path.join(indir, 'all.jsonlist')

    tokenized = []

    with open(infile) as f:
        lines = f.readlines()
    print(len(lines))

    for line in tqdm(lines):            
        line = json.loads(line)            
        line_id = line['id']

        tokens = line['tokens']
        if alt_spellings:
            tokens = [replacements[t] if t in replacements else t for t in tokens]
        line['tokens'] = tokens

        text = ' '.join(tokens)
        # remove ## sybmols
        text = re.sub('##', '', text)        

        # retokenize to get proper tokenization of words that have been replaced
        encoded = tokenizer.encode(text, add_special_tokens=False)
        raw_pieces = tokenizer.convert_ids_to_tokens(encoded)
        rejoined_pieces = []
        
        # rejoin into concatenated words
        if len(raw_pieces) == 0:
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

            outline = {'id': line_id, 'tokens': rejoined_pieces, 'corpus': 'constitution'}
            tokenized.append(outline)
    
    print(len(tokenized), len(lines))

    with open(os.path.join(outdir, 'all.jsonlist'), 'w') as f:
        for line in tokenized:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
