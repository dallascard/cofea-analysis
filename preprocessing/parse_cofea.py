import os
import re
import json
from glob import glob
from optparse import OptionParser

import spacy
import numpy as np
from tqdm import tqdm


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Source to import [founders|statutes|farrands|elliots|hein|evans] (None=all): default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    source = options.source
    spacy_max = 900000  # slightly below max to avoid close calls

    indir = os.path.join(basedir, 'clean')
    parsed_dir = os.path.join(basedir, 'parsed')

    if not os.path.exists(parsed_dir):
        os.makedirs(parsed_dir)

    if source is not None:
        files = [os.path.join(indir, source + '.jsonlist')]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    print("Loading spacy")
    nlp = spacy.load("en_core_web_sm", disable=['ner'])

    for infile in files:
        basename = os.path.basename(infile)
        print(infile)
        with open(infile) as f:
            lines = f.readlines()
        print(len(lines))

        parsed_lines = []

        for line in tqdm(lines):
            line = json.loads(line)
            line_id = line['id']
            # drop the header
            text = line['body']

            if basename == 'evans.jsonlist':
                parts = text.split('\n\n')
            elif basename == 'hein.jsonlist':
                text_len = len(text)
                if text_len > spacy_max:   # spaCy max length
                    tokens = text.split()
                    n_tokens = len(tokens)
                    n_parts = int(np.ceil(text_len / spacy_max))
                    part_len = int(np.ceil(n_tokens / n_parts))
                    print("Splitting {:d} tokens into {:d} parts of {:d} tokens".format(n_tokens, n_parts, part_len))
                    chunks = [tokens[i:i + part_len] for i in range(0, n_tokens, part_len)]
                    print([len(c) for c in chunks])
                    parts = [' '.join(chunk) for chunk in chunks]
                    print([len(p) for p in parts])
                else:
                    parts = [text]
            else:
                parts = [text]

            for p_i, part in enumerate(parts):
                # replace newlines and long spaces
                part = re.sub(r'\s+', ' ', part).strip()
                # parse the text
                if len(part) > 0:
                    parsed = nlp(part)

                    part_id = line_id + '_' + str(p_i).zfill(5)

                    # collect features to be saved
                    sents = []
                    tokens = []
                    lemmas = []
                    tags = []
                    whitespace = []
                    for sent in parsed.sents:
                        sents.append(sent.text.strip())
                        tokens.append([token.text for token in sent])
                        lemmas.append([token.lemma_ for token in sent])
                        tags.append([token.tag_ for token in sent])
                        whitespace.append([token.whitespace_ for token in sent])

                    # save parsed and tokenized representations separtely (for faster reading later)
                    parsed_lines.append({'id': part_id, 'sents': sents, 'tokens': tokens, 'spaces': whitespace, 'lemmas': lemmas, 'tags': tags})

        outfile = os.path.join(parsed_dir, basename)
        with open(outfile, 'w') as fo:
            for line in parsed_lines:
                fo.write(json.dumps(line) + '\n')

    print("Done!")


if __name__ == '__main__':
    main()
