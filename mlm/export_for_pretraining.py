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
    parser.add_option('--coca-dir', type=str, default=None,
                      help='COCA dir (optional): default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--output-subdir', type=str, default='mlm_pretraining',
                      help='Subdirectory for output: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Source(s) to use (comma-separated) [founders|statutes|farrands|elliots|hein|evans|news] (None=all): default=%default')
    parser.add_option('--pre', type=int, default=None,
                      help='Only include documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=None,
                      help='Only include documents from after this year: default=%default')
    parser.add_option('--val-frac', type=float, default=0.05,
                      help='Fraction to use for validation: default=%default')
    parser.add_option('--alt-spellings', action="store_true", default=False,
                      help="Do replacements for alt spellings: default=%default")
    parser.add_option('--max-len', type=int, default=512,
                      help='Maximum number of tokens per line: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    coca_dir = options.coca_dir
    model = options.model
    output_subdir = options.output_subdir
    source = options.source
    pre_year = options.pre
    post_year = options.post  
    val_frac = options.val_frac
    alt_spellings = options.alt_spellings    
    max_len = options.max_len
    seed = options.seed    
    np.random.seed(seed)

    model_name = get_model_name(model)

    # get the tokenized forms of mispelling replacements
    replacements = get_replacements(tokenized=True)
    
    df = pd.read_csv(os.path.join('common', 'hein_years.csv'), header=0, index_col=None)
    corrected_hein_years_by_id = dict(zip(df['ID'].values, df['inferred'].values))

    if post_year is None:
        start = -10000
    else:
        start = post_year + 1
    if pre_year is None:
        end = 10000
    else:
        end = pre_year - 1

    indir = os.path.join(basedir, 'tokenized_' + model_name)
    if not os.path.exists(indir):
        raise FileNotFoundError("Could not find tokenized data in", indir)
    outdir = os.path.join(basedir, output_subdir)

    outdir += '_' + model    
    if pre_year is not None and post_year is not None:
        outdir += '_' + str(post_year+1) + '-' + str(pre_year-1)
    elif pre_year is not None:
        outdir += '_pre-' + str(pre_year)
    elif post_year is not None:
        outdir += '_post-' + str(post_year)
    outdir += '-val' + str(val_frac)
    if not alt_spellings:
        outdir += '_no-alt'
    if coca_dir is not None:
        outdir += '_plus_coca'

    print("Using outdir", outdir)

    if not os.path.exists(outdir):        
        os.makedirs(outdir)

    if source is not None:
        sources = source.split(',')
        files = [os.path.join(indir, s + '.jsonlist') for s in sources]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    coca_files = []
    if coca_dir is not None:
        coca_files.extend(sorted(glob(os.path.join(coca_dir, 'tokenized_' + model_name, '*.jsonlist'))))

    outlines = []
    outlines_val = []
    piece_counts = []

    for f_i, infile in enumerate(files + coca_files):
        basename = os.path.basename(infile)
        print(infile)
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
                # Overwrite Hein document years with inferred values (only affects a handful for filtering)
                if basename == 'hein.jsonlist' and line_id in corrected_hein_years_by_id:
                    year = corrected_hein_years_by_id[line_id]

            if start <= year <= end or f_i >= len(files):
                tokens = line['tokens']      
                if alt_spellings:
                    tokens = [replacements[t] if t in replacements else t for t in tokens]
                text = ' '.join(tokens)
                temp = re.sub('##', ' ##', text)
                pieces = temp.split()
                n_pieces = len(pieces)
                
                # if necessary, break the line up into chunks
                if n_pieces > max_len:
                    # get the number of lines based on the number of pieces
                    n_lines = int(np.ceil(n_pieces / max_len))
                    # divide the number of pieces evenly across those lines
                    n_pieces_per_line = int(np.round(n_pieces / n_lines))

                    lines_to_write = []
                    output_pieces = []
                    for p_i, piece in enumerate(pieces):
                        # if this is the last piece, append it and save the line
                        if p_i == len(pieces) - 1:
                            output_pieces.append(piece)
                            piece_counts.append(len(output_pieces))
                            line_to_write = ' '.join(output_pieces)                            
                            line_to_write = re.sub(' ##', '', line_to_write)
                            lines_to_write.append(line_to_write)

                            if len(output_pieces) >= 573:
                                print(basename, line_id, n_pieces, n_lines, n_pieces_per_line, p_i)
                                print(output_pieces)

                        # if we have got enough pieces for a line, save it, unless this token is a continuation
                        elif len(output_pieces) >= n_pieces_per_line and not piece.startswith('##'):
                            piece_counts.append(len(output_pieces))
                            line_to_write = ' '.join(output_pieces)
                            line_to_write = re.sub(' ##', '', line_to_write)
                            lines_to_write.append(line_to_write)

                            if len(output_pieces) >= 573:
                                print(basename, line_id, n_pieces, n_lines, n_pieces_per_line, p_i)
                                print(output_pieces)

                            output_pieces = [piece]                            
                        
                        # if so, just add it, and try again on the next piece
                        else:
                            output_pieces.append(piece)
                            
                    for line_to_write in lines_to_write:
                        if val_frac > 0 and np.random.rand() <= val_frac:
                            outlines_val.append(line_to_write)
                        else:
                            outlines.append(line_to_write)

                else:
                    piece_counts.append(n_pieces)
                    text = re.sub('##', '', text)        

                    if val_frac > 0 and np.random.rand() <= val_frac:
                        outlines_val.append(text.strip())
                    else:
                        outlines.append(text.strip())


        print(len(outlines), len(outlines_val))
        print()

    print("Max number of pieces:", np.max(piece_counts))
    print("Number of chunks over max_len pieces:", np.sum(np.array(piece_counts) > max_len))

    if len(outlines) > 0:
        if source is not None:
            outfile = os.path.join(outdir, source + '_raw_train.txt')
        else:
            outfile = os.path.join(outdir, 'all_raw_train.txt')

        order = np.arange(len(outlines))
        np.random.shuffle(order)
        with open(outfile, 'w') as f:
            for i in order:
                f.write(outlines[i] + '\n')

    if len(outlines_val) > 0:
        if source is not None:
            outfile = os.path.join(outdir, source + '_raw_val.txt')
        else:
            outfile = os.path.join(outdir, 'all_raw_val.txt')

        order = np.arange(len(outlines_val))
        np.random.shuffle(order)
        with open(outfile, 'w') as f:
            for i in order:
                f.write(outlines_val[i] + '\n')

    print("Output written to", outdir)


if __name__ == '__main__':
    main()
