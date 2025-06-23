import os
import re
import json
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.bigrams import concat_ngrams
from common.alt_spellings import get_replacements


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--cofea-dir', type=str, default='/data/dalc/COFEA/',
                      help='COFEA dir: default=%default')
    parser.add_option('--const-dir', type=str, default='/data/dalc/constitution/',
                      help='Basedir: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='BERT model: default=%default')
    parser.add_option('--pre', type=int, default=1801,
                      help='Only include documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=1759,
                      help='Only include documents from after this year: default=%default')

    (options, args) = parser.parse_args()

    cofea_dir = options.cofea_dir
    const_dir = options.const_dir  
    model = options.model
    pre = options.pre
    post = options.post

    replacements = get_replacements()

    outdir = os.path.join('plotting', 'plot_data')
    suffix = ''
    if post is not None:
        suffix += '_post-' + str(post)
    if pre is not None:
        suffix += '_pre-' + str(pre)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    infile = os.path.join(const_dir, 'tokenized_' + model, 'all.jsonlist')

    print("Loading constitution")
    const_terms = set()
    with open(infile) as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        tokens = line['tokens']
        tokens = [re.sub('##', '', t) for t in tokens]
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [replacements[t] if t in replacements else t for t in tokens]
        tokens = concat_ngrams(tokens)        
        const_terms.update(tokens)
    const_terms = sorted(const_terms)
    print(len(const_terms), 'unique terms in the constitution')

    df = pd.read_csv(os.path.join('common', 'hein_years.csv'), header=0, index_col=None)
    corrected_hein_years_by_id = dict(zip(df['ID'].values, df['inferred'].values))

    counts_by_subset = defaultdict(Counter)
    freqs_by_subset = defaultdict(dict)

    for source in ['evans', 'founders', 'hein', 'statutes', 'elliots', 'founders', 'news']:
        if source == 'evans':
            subset = 'Popular'
        elif source == 'news':
            subset = 'Popular'
        elif source == 'founders':
            subset = 'Founders'
        else:
            subset = 'Legal'

        print(source, subset)

        infile = os.path.join(cofea_dir, 'tokenized_' + model, source + '.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            line_id = line['id']
            year = line['year']
            year = line['year']            
            if source == 'hein':
                year = corrected_hein_years_by_id[line_id]

            skip = False
            if pre is not None and year >= pre:
                skip = True
            if post is not None and year <= post:
                skip = True
            if not skip:
                tokens = line['tokens']
                tokens = [re.sub('##', '', t) for t in tokens]
                tokens = [replacements[t] if t in replacements else t for t in tokens]
                tokens = concat_ngrams(tokens)

                if source == 'hein':
                    if line['genre'] == 'Legal':
                        counts_by_subset[subset].update(tokens)
                else:
                    counts_by_subset[subset].update(tokens)

    for subset in counts_by_subset:
        total = sum(counts_by_subset[subset].values())
        for term in const_terms:
            freqs_by_subset[term][subset] = counts_by_subset[subset][term] / total

    df = pd.DataFrame()
    df['terms'] = const_terms
    rel_freqs = np.zeros([len(const_terms), 3])

    for term_i, term in enumerate(const_terms):
        total = sum(freqs_by_subset[term].values())
        rel_freqs[term_i, :] = [freqs_by_subset[term]['Popular']/total, freqs_by_subset[term]['Founders']/total, freqs_by_subset[term]['Legal']/total]
        if total > 0:
            print(term, '{:.6f} {:.6f} {:.6f}'.format(freqs_by_subset[term]['Popular']/total, freqs_by_subset[term]['Founders']/total, freqs_by_subset[term]['Legal']/total))
        else:
            print(term, 0, 0, 0)
        
    for subset in ['Popular', 'Founders', 'Legal']:
        df[subset + '_count'] = [counts_by_subset[subset][t] for t in const_terms]
    df['Total_count'] = [counts_by_subset['Popular'][t] + counts_by_subset['Founders'][t] + counts_by_subset['Legal'][t] for t in const_terms]

    df['Popular'] = rel_freqs[:, 0]
    df['Founders'] = rel_freqs[:, 1]
    df['Legal'] = rel_freqs[:, 2]

    outfile = os.path.join(outdir, 'rel_freqs' + suffix + '.csv')
    df.to_csv(outfile)


if __name__ == '__main__':
    main()
