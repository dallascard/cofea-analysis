import os
import re
import json
from glob import glob
from optparse import OptionParser

from tqdm import tqdm

import spacy


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='/data/dalc/COFEA/tokenized_bert-large-uncased/',
                      help='Directory with documents tokenized by bert: default=%default')
    parser.add_option('--outdir', type=str, default='/data/dalc/COFEA/tokenized_bert-large-uncased-lemmatized/',
                      help='Outdir: default=%default')

    (options, args) = parser.parse_args()

    nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])
    nlp.max_length = 10000000

    indir = options.indir
    outdir = options.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    files = glob(os.path.join(indir, '*.jsonlist'))
    for infile in files:
        print(infile)
        outlines = []
        with open(infile) as f:
            lines = f.readlines()
        for line_i, line in enumerate(tqdm(lines)):
            line = json.loads(line)
            tokens = line['tokens']
            tokens = [re.sub('##', '', t) for t in tokens]

            text = ' '.join(tokens)
            processed = nlp(text)
            tokens = [token.lemma_ for token in processed]

            line['tokens'] = tokens
            outlines.append(line)

        outfile = os.path.join(outdir, os.path.basename(infile))
        with open(outfile, 'w') as f:
            for line in outlines:
                f.write(json.dumps(line) + '\n')

    print("Done!")
    

if __name__ == '__main__':
    main()
