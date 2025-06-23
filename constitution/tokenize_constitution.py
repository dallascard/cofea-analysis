import os
import re
import json
from optparse import OptionParser

from tqdm import tqdm
from transformers import BertTokenizer


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/constitution',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model_name_or_path = options.model

    output_subdir = 'tokenized_' + model_name_or_path
    outdir = os.path.join(basedir, output_subdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    infile = os.path.join(basedir, 'clean', 'constitution.jsonlist')
    outfile = os.path.join(outdir, 'all.jsonlist')

    print("Loading model")
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

    outlines = []    
    
    with open(infile) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        line = json.loads(line)
        line_id = line['id']
        text = line['text'].strip()
        
        # replace double pound signs
        text = re.sub(r'#+', '#', text)

        if len(text) == 0:
            print("Empty text in", line_id)
            print(line)
        else:
            # convert to tokens using tokenizer
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
                outlines.append(outline)

    with open(outfile, 'w') as fo:
        for line in outlines:
            fo.write(json.dumps(line) + '\n')

    print("Output written to", outfile)


if __name__ == '__main__':
    main()
