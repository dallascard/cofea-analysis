import os
import re
import json
from glob import glob
from collections import Counter
from optparse import OptionParser

import pandas as pd
from tqdm import tqdm

from common.bigrams import concat_ngrams
from common.alt_spellings import get_replacements


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--const-dir', type=str, default='/data/dalc/constitution/',
                      help='Base directory: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Source to import [founders|statutes|farrands|elliots|hein|evans] (None=all): default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Base directory: default=%default')
    parser.add_option('--pre', type=int, default=None,
                      help='Only count documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=None,
                      help='Only count documents from after this year: default=%default')
    parser.add_option('--ignore-years', action="store_true", default=False,
                      help="Ignore years (e.g., for COCA): default=%default")
    parser.add_option('--legal-only', action="store_true", default=False,
                      help="Only count legal documents: default=%default")
    parser.add_option('--non-legal', action="store_true", default=False,
                      help="Only count non-legal documents: default=%default")
    parser.add_option('--no-correction', action="store_true", default=False,
                      help="Do not use corrected Hien years: default=%default")
    parser.add_option('--test-term', type=str, default=None,
                      help='term to print occurrences of: default=%default')


    (options, args) = parser.parse_args()

    basedir = options.basedir
    const_dir = options.const_dir
    source = options.source
    model = options.model
    pre_year = options.pre
    post_year = options.post  
    ignore_years = options.ignore_years
    legal_only = options.legal_only
    non_legal = options.non_legal
    no_correction = options.no_correction
    test_term = options.test_term

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
    if no_correction:
        suffix += '_non-corrected-years'

    if post_year is None:
        start = -10000
    else:
        start = post_year + 1
    if pre_year is None:
        end = 10000
    else:
        end = pre_year - 1

    indir = os.path.join(basedir, 'tokenized_' + model)
    if not os.path.exists(indir):
        raise FileNotFoundError(indir, "not found")
    outdir = os.path.join(indir, 'counts')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    replacements = get_replacements()

    print("Loading constitution")
    const_terms = set()
    infile = os.path.join(const_dir, 'tokenized_' + model, 'all.jsonlist')
    with open(infile) as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        tokens = line['tokens']
        tokens = [re.sub('##', '', t) for t in tokens]
        tokens = [replacements[t] if t in replacements else t for t in tokens]
        tokens = concat_ngrams(tokens)
        const_terms.update(tokens)
    const_terms = sorted(const_terms)
    print(len(const_terms), 'unique terms in the constitution')

    if source is not None:
        files = [os.path.join(indir, source + '.jsonlist')]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    df = pd.read_csv(os.path.join('common', 'hein_years.csv'), header=0, index_col=None)
    corrected_hein_years_by_id = dict(zip(df['ID'].values, df['inferred'].values))

    token_counter = Counter({t: 0 for t in const_terms})

    for infile in files:
        print(infile)
        basename = os.path.basename(infile)
        subset = basename[:-9]
        with open(infile) as f:
            lines = f.readlines()
        print(len(lines))

        for line in tqdm(lines):            
            line = json.loads(line)    
            line_id = line['id']
            
            if 'year' not in line:
                assert pre_year is None
                assert post_year is None
                year = 0
            else:
                year = line['year']            
                if basename == 'hein.jsonlist' and not no_correction:
                    year = corrected_hein_years_by_id[line_id]

            # allow counting only legal or non-legal documents
            line['legal'] = False
            skip = False
            if subset in {'statutes', 'farrands', 'elliots'}:
                line['legal'] = True
            elif subset == 'hein':
                if line['genre'] == 'Legal':
                    line['legal'] = True
                else:
                    skip = True

            if legal_only and not line['legal']:
                skip = True
            if non_legal and line['legal']:
                skip = True

            if (ignore_years or start <= year <= end) and not skip:
                tokens = line['tokens']
                tokens = [re.sub('##', '', t) for t in tokens]
                tokens = [replacements[t] if t in replacements else t for t in tokens]
                tokens = concat_ngrams(tokens)
                token_set = set(tokens)
                if test_term is not None and test_term in token_set:
                    print(test_term, "found in", line_id)
                token_counter.update(tokens)

    if len(token_counter) > 0:
        for t, c in token_counter.most_common(n=3):
            print(t, c)

        token_counter_sorted = {t: c for t, c in token_counter.most_common()}

        if source is not None:
            outfile = os.path.join(outdir, source + '_token_counts' + suffix + '.json')
        else:
            outfile = os.path.join(outdir, 'token_counts' + suffix + '.json')
        with open(outfile, 'w') as fo:
            json.dump(token_counter_sorted, fo, indent=2)

        const_subset_sorted = {t: c for t, c in token_counter.most_common() if t in const_terms}

        if source is not None:
            outfile = os.path.join(outdir, source + '_token_counts_const' + suffix + '.json')
        else:
            outfile = os.path.join(outdir, 'token_counts_const' + suffix + '.json')
        with open(outfile, 'w') as fo:
            json.dump(const_subset_sorted, fo, indent=2)


if __name__ == '__main__':
    main()
