import os
import copy
import json
from optparse import OptionParser

import pandas as pd


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='plotting/plot_data/mlm_early_vs_modern_1760-1800/jsd_scores_targets.csv',
                      help='First model file (base): default=%default')
    parser.add_option('--outfile', type=str, default='plotting/plots/early_vs_modern.html',
                      help='Second model file (to align): default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outfile = options.outfile

    df = pd.read_csv(infile, header=0)
    
    df = df.drop(columns=['neighbours', 'scaled_jsd'])

    df.sort_values(by='jsd', ascending=False, inplace=True)

    terms = df['term'].values
    jsd = df['jsd'].values
    count_early = df['count_early'].values
    count_modern = df['count_modern'].values

    outlines = []
    outlines.extend([
        '<div id="tableContainer" class="tableContainer">',
        '<table border="0" cellpadding="0" cellspacing="0" width="100%" class="scrollTable">',
        '<thead class="fixedHeader">',
        '   <tr>',
        '       <th><a href="#">Term</a></th>',
        '       <th><a href="#">JSD</a></th>',
        '       <th><a href="#">Early Count</a></th>',
        '       <th><a href="#">Modern Count</a></th>',        
        '   </tr>',
        '</thead>',
        '<tbody class="scrollContent">',
    ])

    for i, term in enumerate(terms):
        outlines.extend([
            '   <tr>',
            f'       <td>{term}</td>',
            f'       <td>{jsd[i]:.3f}</td>',
            f'       <td>{count_early[i]}</td>',
            f'       <td>{count_modern[i]}</td>',
            '   </tr>'
        ])

    outlines.extend([
        '</tbody>',
        '</table>',
        '</div>'
    ])

    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
