import os
import json
from optparse import OptionParser

from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/u/scr/nlp/data/ECCO_TCP/',
                      help='Base directory: default=%default')
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')

    (options, args) = parser.parse_args()

    ecco_dir = options.basedir
    model_type = options.model_type
    model_name_or_path = options.model
    tokenizer = options.tokenizer

    indir = os.path.join(ecco_dir, 'clean')
    outdir = os.path.join(ecco_dir, 'tokenized_' + model_name_or_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading model")
    if model_type == 'bert':
        tokenizer_class = BertTokenizer
    elif model_type == 'roberta':
        tokenizer_class = RobertaTokenizer
    else:
        raise ValueError("Model type not recognized")

    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    printed_empty = False

    infile = os.path.join(indir, 'all.jsonlist')

    empty_docs = 0
    emtpy_after_tokenization = 0
    outlines = []
    print(infile)
    with open(infile) as f:
        lines = f.readlines()
    print(len(lines))

    max_n_pieces = 0
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
                    print("Empty text in", line_id)
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
                    outline = {'id': line_id, 'tokens': rejoined_pieces}
                    for field in ['year', 'title', 'author']:
                        if field in line:
                            outline[field] = line[field]

                    outlines.append(outline)

    print(len(lines), empty_docs, emtpy_after_tokenization, len(outlines), len(lines) - empty_docs -len(outlines))
    outfile = os.path.join(outdir, 'all.jsonlist')
    with open(outfile, 'w') as fo:
        for line in outlines:
            fo.write(json.dumps(line) + '\n')
    
    print("Output written to", outfile)


if __name__ == '__main__':
    main()
