import os
import re
import json
import string
from glob import glob
import datetime as dt
from collections import defaultdict, Counter
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--cofea-dir', type=str, default='/data/dalc/COFEA/',
                      help='COFEA dir: default=%default')
    parser.add_option('--const-dir', type=str, default='/data/dalc/constitution/',
                      help='Basedir: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='BERT model: default=%default')
    parser.add_option('--pre', type=int, default=None,
                      help='Only include documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=None,
                      help='Only include documents from after this year: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.cofea_dir

    tokens_by_year_by_source = defaultdict(Counter)
    basedir = '/data/dalc/COFEA/'
    files = sorted(glob(os.path.join(basedir, 'tokenized_bert-large-uncased', '*.jsonlist')))
    for infile in files:
        print(infile)
        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        basename = os.path.basename(infile)
        for line in tqdm(lines):
            if basename[:-9] == 'founders':
                source = 'Founders: ' + line['collection']
            elif basename[:-9] == 'news':
                source = 'The Pennsylvania Gazette'
            elif basename[:-9] == 'hein':
                source = 'Hein'
            elif basename[:-9] == 'elliots':
                source = 'Elliot’s Debates'
            elif basename[:-9] == 'statutes':
                source = 'United States Statutes at Large'
            elif basename[:-9] == 'evans':
                source = 'Evans Early American Imprints'
            elif basename[:-9] == 'farrands':
                source = "Farrand's Records"
            else:
                print(infile)
            year = line['year']
            tokens_by_year_by_source[source][year] += len(line['tokens'])
            if basename[:-9] == 'founders':
                tokens_by_year_by_source['Founders'][year] += len(line['tokens'])

    with open('plotting/plot_data/tokens_by_year_by_source.json', 'w') as f:
        json.dump(tokens_by_year_by_source, f, indent=2)

    source_vector = []
    detail_vector = []
    year_vector = []
    count_vector = []

    sources = sorted(tokens_by_year_by_source)
    for i, source in enumerate(sources):
        if source.startswith('Founders:'):
            pass
        elif source.startswith('The P'):
            pass
        else:
            name = source
            years = sorted(tokens_by_year_by_source[source])
            counts = [tokens_by_year_by_source[source][y] for y in years]
            year_chunks = [[years[0]]]
            count_chunks = [[counts[0]]]
            for year_i, year in enumerate(years[1:]):
                if year == year_chunks[-1][-1] + 1:
                    year_chunks[-1].append(year)
                    count_chunks[-1].append(counts[year_i+1])
                else:
                    year_chunks.append([year])
                    count_chunks.append([counts[year_i+1]])        
            for chunk_i, year_chunk in enumerate(year_chunks):
                count_chunk = count_chunks[chunk_i]
                year_vector.extend(year_chunk)
                count_vector.extend(count_chunk)
                source_vector.extend([source]*len(year_chunk))
                detail_vector.extend([source + str(chunk_i)]*len(year_chunk))
                
            
    df_counts = pd.DataFrame({'Year': year_vector, 'Count': count_vector, 'Source': source_vector, 'Detail': detail_vector})
    df_counts.head()

    df_counts.to_csv('plotting/plot_data/tokens_by_year_by_source.csv', index=False)
    print('Wrote plotting/plot_data/tokens_by_year_by_source.csv')
    print('Wrote plotting/plot_data/tokens_by_year_by_source.json')
    print('Done!')


if __name__ == '__main__':
    main()
