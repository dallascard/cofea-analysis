import os
import json
from optparse import OptionParser
from collections import defaultdict

import pandas as pd

from common.misc import convert_hyphens


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/constitution',
                      help='basedir: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    outdir = os.path.join(basedir, 'clean')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    df = pd.read_csv(os.path.join('constitution', 'US_constitution.tsv'), header=0, index_col=None, sep='\t')
    print(df.head())

    ids = df['id'].values
    parts = df['Part'].values
    sections = df['Section'].values
    texts = df['Text'].values
    years = df['Year'].values

    # Convert the document to texts by section (combining subsections)
    text_by_section = defaultdict(list)
    sections_in_order = []
    for i, paragraph_id in enumerate(ids):
        text = texts[i]
        year = years[i]
        part = parts[i]
        section = sections[i]

        # Remove hyphens
        text, _ = convert_hyphens(text)        
        text = text.strip()

        # only keep the first ten amendments (the bill of rights)
        if year < 1792:
            section = part + ' ' + 'section' + ' ' + str(section)
            if section not in sections_in_order:
                sections_in_order.append(section)

            text_by_section[section].append(text)

    # Combine the texts for each section
    outlines = []
    for section in sections_in_order:
        text = ' '.join(text_by_section[section])
        # re-attach the years
        if section.startswith('Amendment'):
            year = 1791
        else:
            year = 1789
        outline = {'id': section, 'year': year, 'text': text}
        outlines.append(outline)
        
    outfile = os.path.join(outdir, 'constitution.jsonlist')
    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    print("Output written to", outfile)

if __name__ == '__main__':
    main()
