import os
import json
from glob import glob
from optparse import OptionParser

import pandas as pd


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/ECCO_TCP/standardized/',
                      help='Base directory: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    outdir = os.path.join(basedir, 'clean')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    metadata_file = os.path.join(basedir, 'ECCOTCP.csv')
    df = pd.read_csv(metadata_file, header=0, index_col=None)

    years = df['Date'].values
    titles = df['Title'].values
    authors = df['Author'].values
    doc_ids = df['TCP'].values
    doc_index = dict(zip(doc_ids, range(len(doc_ids))))

    files = sorted(glob(os.path.join(basedir, '*', '*.txt')))

    outlines = []
    for infile in files:
        with open(infile) as f:
            text = f.read()
        
        doc_id = os.path.splitext(os.path.basename(infile))[0]
        index = doc_index[doc_id]

        outlines.append({'id': doc_id,
                         'year': int(years[index]),
                         'author': authors[index],
                         'title': titles[index],
                         'text': text
                         })

    outfile = os.path.join(outdir, 'all.jsonlist')

    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    print("Output written to", outfile)


if __name__ == '__main__':
    main()
