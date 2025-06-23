import os
import re
import json
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.alt_spellings import get_replacements
from common.misc import get_model_name


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--coca-dir', type=str, default='/data/dalc/COCA/',
                      help='COCA directory: default=%default')
    parser.add_option('--output-subdir', type=str, default='mlm_early_vs_modern',
                      help='Output subdirectory: default=%default')    
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Source(s) to use (comma-separated) [founders|statutes|farrands|elliots|hein|evans|news] (None=all): default=%default')
    parser.add_option('--pre', type=int, default=None,
                      help='Only include early documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=None,
                      help='Only include early documents from after this year: default=%default')
    parser.add_option('--alt-spellings', action="store_true", default=False,
                      help="Do replacements for alt spellings: default=%default")
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    coca_dir = options.coca_dir
    output_subdir = options.output_subdir
    model = options.model
    source = options.source
    pre_year = options.pre
    post_year = options.post  
    alt_spellings = options.alt_spellings
    seed = options.seed    
    np.random.seed(seed)

    model_name = get_model_name(model)

    # get the tokenized forms of mispelling replacements
    replacements = get_replacements(tokenized=True)

    suffix = ''
    if pre_year is not None and post_year is not None:
        suffix += '_' + str(post_year+1) + '-' + str(pre_year-1)
    elif pre_year is not None:
        suffix += '_pre-' + str(pre_year)
    elif post_year is not None:
        suffix += '_post-' + str(post_year)
    if not alt_spellings:
        suffix += '_no-alt'

    if post_year is None:
        start = 1
    else:
        start = post_year + 1
    if pre_year is None:
        end = 10000
    else:
        end = pre_year - 1

    indir = os.path.join(basedir, 'tokenized_' + model_name)
    print("Using data from", indir)
    outdir = os.path.join(basedir, output_subdir + suffix)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if source is not None:
        sources = source.split(',')
        files = [os.path.join(indir, s + '.jsonlist') for s in sources]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    coca_files = set()
    if coca_dir is not None:
        coca_files.update(sorted(glob(os.path.join(coca_dir, 'tokenized_' + model_name, '*.jsonlist'))))
        if len(coca_files) == 0:
            print("No COCA files found in", os.path.join(coca_dir, 'tokenized_' + model_name))
            raise ValueError("No COCA files found")

    df = pd.read_csv(os.path.join('common', 'hein_years.csv'), header=0, index_col=None)
    corrected_hein_years_by_id = dict(zip(df['ID'].values, df['inferred'].values))

    outlines = []
    tokenized = []
    for infile in files + sorted(coca_files):
        if infile in coca_files:
            corpus = 'modern'
        else:
            corpus = 'early'
        basename = os.path.basename(infile)
        print(infile)
        with open(infile) as f:
            lines = f.readlines()
        print(len(lines))

        for line in tqdm(lines):            
            line = json.loads(line)            
            line_id = line['id']
            line['corpus'] = corpus

            if 'year' not in line:
                assert pre_year is None
                assert post_year is None
                year = 0
            else:
                year = line['year']
                if basename == 'hein.jsonlist' and line_id in corrected_hein_years_by_id:
                    if year != corrected_hein_years_by_id[line_id]:
                        year = corrected_hein_years_by_id[line_id]

            if (start <= year <= end) or corpus == 'modern':
                tokens = line['tokens']
                if alt_spellings:
                    tokens = [replacements[t] if t in replacements else t for t in tokens]
                line['tokens'] = tokens
                line['year'] = int(year)
                text = ' '.join(tokens)
                text = re.sub('##', '', text)        

                outlines.append(text)
                tokenized.append(line)             

        print(len(outlines))
        print()
    
    if len(outlines) > 0:

        order = np.arange(len(outlines))
        np.random.shuffle(order)

        if source is not None:
            outfile = os.path.join(outdir, source + '.jsonlist')
        else:
            outfile = os.path.join(outdir, 'all.jsonlist')

        with open(outfile, 'w') as f:
            for i in order:
                f.write(json.dumps(tokenized[i]) + '\n')

        print("Output written to", outfile)


if __name__ == '__main__':
    main()
