import os
import re
from glob import glob
from optparse import OptionParser

import kenlm
import pandas as pd
from tqdm import tqdm


# Compute characte-level perplexities on each document using kenlm

def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='/Users/dalc/data/COFEA/char_lm_data/',
                      help='Directory with documents tokenized as characters: default=%default')
    parser.add_option('--model-file', type=str, default='/Users/dalc/tools/kenlm/cofea/model.arpa',
                      help='Clean accessible archives directory: default=%default')

    (options, args) = parser.parse_args()

    indir = options.indir
    model_file = options.model_file

    model = kenlm.LanguageModel(model_file)

    files = glob(os.path.join(indir, '*.csv'))
    
    for infile in files:        
        print(infile)
        scores = []
        lengths = []

        with open(infile) as f:
            lines = f.readlines()
        doc_ids = [line.strip() for line in lines] 

        with open(infile.replace('.csv', '.txt')) as f:
            lines = f.readlines()

        assert len(lines) == len(doc_ids)

        for line in tqdm(lines):
            score = model.score(line)
            # get the number of characters in the line
            line = re.sub(r'\s+', '', line)
            line = re.sub(r'<s>', ' ', line)    
            lengths.append(len(line))
            assert len(line) > 0
            scores.append(score / len(line))

        df = pd.DataFrame({'id': doc_ids, 'score': scores, 'length': lengths})

        outfile = os.path.join(indir, os.path.basename(infile).replace('.csv', '_ppl.tsv'))
        df.to_csv(outfile, sep='\t', index=False)


if __name__ == '__main__':
    main()
