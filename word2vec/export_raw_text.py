import os
import re
import json
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.bigrams import concat_ngrams
from common.alt_spellings import get_replacements


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--output-subdir', type=str, default='word2vec',
                      help='Subdirectory for output: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Source(s) to use (comma-separated) [founders|statutes|farrands|elliots|hein|evans|news] (None=all): default=%default')
    parser.add_option('--legal-only', action="store_true", default=False,
                      help='Export legal subset of data (alternative to --source): default=%default')
    parser.add_option('--non-legal', action="store_true", default=False,
                      help='Export non-legal subset of data (Evans and news): default=%default')
    parser.add_option('--pre', type=int, default=None,
                      help='Only include documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=None,
                      help='Only include documents from after this year: default=%default')
    parser.add_option('--val-frac', type=float, default=0.,
                      help='Fraction to use for validation: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--bigrams', action="store_true", default=False,
                      help='Concatenate bigrams: default=%default')
    parser.add_option('--alt-spellings', action="store_true", default=False,
                      help='Replace alt spellings (once they have been found): default=%default')
    parser.add_option('--from-parsed', action="store_true", default=False,
                      help='Use tokenization from spacy rather than bert: default=%default')
    parser.add_option('--lower', action="store_true", default=False,
                      help='Lower case all text: default=%default')
    parser.add_option('--ignore-year', action="store_true", default=False,
                      help='Ignore year: default=%default')


    (options, args) = parser.parse_args()

    basedir = options.basedir
    model_name_or_path = options.model
    output_subdir = options.output_subdir
    source = options.source
    legal_only = options.legal_only
    non_legal = options.non_legal
    pre_year = options.pre
    post_year = options.post  
    val_frac = options.val_frac
    seed = options.seed    
    np.random.seed(seed)
    use_bigrams = options.bigrams
    use_alt_spellings = options.alt_spellings
    from_parsed = options.from_parsed
    lower = options.lower
    ignore_year = options.ignore_year

    replacements = get_replacements()

    if pre_year is not None and post_year is not None:
        suffix = '_' + str(post_year+1) + '-' + str(pre_year-1)
    elif pre_year is not None:
        suffix = '_pre-' + str(pre_year)
    elif post_year is not None:
        suffix = '_post-' + str(post_year)
    else:
        suffix = ''
    if legal_only:
        suffix += '_legal'
    if non_legal:
        suffix += '_non-legal'

    if post_year is None:
        start = 1500
    else:
        start = post_year + 1
    if pre_year is None:
        end = 2100
    else:
        end = pre_year - 1

    if from_parsed:
        indir = os.path.join(basedir, 'parsed')    
    else:
        indir = os.path.join(basedir, 'tokenized_' + model_name_or_path)
    print("Reading from", indir)
    outdir = os.path.join(basedir, output_subdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)        

    if legal_only:
        files = [os.path.join(indir, s + '.jsonlist') for s in ['statutes', 'farrands', 'elliots', 'hein']]
    elif non_legal:
        files = [os.path.join(indir, s + '.jsonlist') for s in ['evans', 'news']]
    elif source is not None:
        sources = source.split(',')
        files = [os.path.join(indir, s + '.jsonlist') for s in sources]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    outlines = []
    outlines_val = []
    for infile in files:

        basename = os.path.basename(infile)
        print(infile)
        with open(infile) as f:
            lines = f.readlines()
        print(len(lines))

        for line in tqdm(lines):            
            line = json.loads(line)    
            
            if basename == 'hein.jsonlist' and legal_only and line['genre'] != 'Legal':
                pass
            else:
                if 'year' not in line:
                    assert pre_year is None
                    assert post_year is None
                    year = 0
                else:
                    year = line['year']
                        
                if start <= year <= end or ignore_year:
                    if from_parsed:
                        tokens = []
                        for sent in line['tokens']:
                            if lower:
                                tokens.extend([t.lower() for t in sent])
                            else:
                                tokens.extend(sent)
                    else:
                        tokens = line['tokens']            
                        tokens = [re.sub('##', '', t) for t in tokens]
                    
                    if use_alt_spellings:
                        tokens = [replacements[t] if t in replacements else t for t in tokens]

                    if use_bigrams:
                        combined = concat_ngrams(tokens)
                    else:
                        combined = tokens
                        
                    if val_frac > 0 and np.random.rand() <= val_frac:
                        outlines_val.append(' '.join(combined))
                    else:
                        outlines.append(' '.join(combined))

        print(len(outlines), len(outlines_val))
        print()

    if len(outlines) > 0:
        if source is not None:
            outfile = os.path.join(outdir, source + '_raw_train' + suffix + '.txt')
        else:
            outfile = os.path.join(outdir, 'all_raw_train' + suffix + '.txt')

        order = np.arange(len(outlines))
        np.random.shuffle(order)
        with open(outfile, 'w') as f:
            for i in order:
                f.write(outlines[i] + '\n')

    if len(outlines_val) > 0:
        if source is not None:
            outfile = os.path.join(outdir, source + '_raw_val' + suffix + '.txt')
        else:
            outfile = os.path.join(outdir, 'all_raw_val' + suffix + '.txt')

        order = np.arange(len(outlines_val))
        np.random.shuffle(order)
        with open(outfile, 'w') as f:
            for i in order:
                f.write(outlines_val[i] + '\n')


if __name__ == '__main__':
    main()
