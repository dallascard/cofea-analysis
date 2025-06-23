import os
import re
import json
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--aa-dir', type=str, default='/data/dalc/accessible_archives/clean/',
                      help='Clean accessible archives directory: default=%default')
    parser.add_option('--ecco-dir', type=str, default='/data/dalc/ECCO_TCP/standardized/clean/',
                      help='Clean ECCO TCP dir: default=%default')
    parser.add_option('--output-subdir', type=str, default='char_lm_data',
                      help='Subdirectory for output: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Source(s) to use (comma-separated) [founders|statutes|farrands|elliots|hein|evans|news] (None=all): default=%default')
    parser.add_option('--pre', type=int, default=None,
                      help='Only include documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=None,
                      help='Only include documents from after this year: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    aa_dir = options.aa_dir
    ecco_dir = options.ecco_dir
    output_subdir = options.output_subdir
    source = options.source
    seed = options.seed    
    np.random.seed(seed)

    indir = os.path.join(basedir, 'clean')
    outdir = os.path.join(basedir, output_subdir)
    print("Using outdir", outdir)

    if not os.path.exists(outdir):        
        os.makedirs(outdir)

    print("Loading data")
    ecco_file = os.path.join(ecco_dir, 'all.jsonlist')
    with open(ecco_file) as f:
        ecco_lines = f.readlines()

    aa_file = os.path.join(aa_dir, 'news.jsonlist')
    with open(aa_file) as f:
        aa_lines = f.readlines()

    bg_lines = ecco_lines + aa_lines
    np.random.shuffle(bg_lines)

    outlines = []

    for line in tqdm(bg_lines):            
        line = json.loads(line)            
        line_id = line['id']
        text = line['text']
        output_text = convert_to_char(text)
        outlines.append(output_text)

    print("Saving data")
    with open(os.path.join(outdir, 'bg.txt'), 'w') as f:
        for line in outlines:
            f.write(line + '\n')

    if source is not None:
        sources = source.split(',')
        files = [os.path.join(indir, s + '.jsonlist') for s in sources]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    for infile in files:
        basename = os.path.basename(infile)
        print(infile)
        with open(infile) as f:
            lines = f.readlines()
        
        outlines = []
        ids = []
        for line in tqdm(lines):            
            line = json.loads(line)            
            line_id = line['id']
            text = line['body']
            output_text = convert_to_char(text)
            outlines.append(output_text)
            ids.append(line_id)

        print("Saving data")
        with open(os.path.join(outdir, basename.replace('.jsonlist', '.txt')), 'w') as f:
            for line in outlines:
                f.write(line + '\n')

        with open(os.path.join(outdir, basename.replace('.jsonlist', '.csv')), 'w') as f:
            for line in ids:
                f.write(line + '\n')

    print("Done!")


def convert_to_char(text):
    # replace all whitespace with single spaces
    text = re.sub(r'\s+', ' ', text)

    # convert to characters and replace spaces with <space>
    characters = [t if t != ' ' else '<sp>' for t in text]

    # rejoin all characters, separated by spaces
    return ' '.join(characters)


if __name__ == '__main__':
    main()
