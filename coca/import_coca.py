import os
import re
import json
from glob import glob
from collections import Counter
from optparse import OptionParser

from tqdm import tqdm

from common.misc import convert_hyphens


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--coca-dir', type=str, default='/data/dalc/COCA/',
                      help='Directory containing "raw": default=%default')
    parser.add_option('--basedir', type=str, default='/data/dalc/COCA/',
                      help='Base directory: default=%default')
    parser.add_option('--include-spoken', action="store_true", default=False,
                      help='Include text_spoken_kde: default=%default')

    (options, args) = parser.parse_args()

    coca_dir = options.coca_dir
    basedir = options.basedir
    include_spoken = options.include_spoken

    raw_dir = os.path.join(coca_dir, 'raw')
    outdir = os.path.join(basedir, 'clean')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    html_tags = {'<p>', '</p>',
                 '<P>', '</P>', '<P/>',
                 '<u>', '</u>',
                 '<U>', '</U>'
                 '<>',
                 '<br>',
                 '<sc>', '</sc>',
                 '<h>,'
                 '<H3>', '</H3>',
                 '<H4>', '</H4>'}

    html_counter = Counter()
    chunk_lengths = Counter()
    replacement_counter = Counter()
    if include_spoken:
        subdirs = ['text_academic_rpe', 'text_fiction_awq', 'text_magazine_qch', 'text_newspaper_lsp', 'text_spoken_kde']
    else:
        subdirs = ['text_academic_rpe', 'text_fiction_awq', 'text_magazine_qch', 'text_newspaper_lsp']
    for subdir in subdirs:
        print(subdir)
        outlines = []
        genre = subdir.split('_')[1]
        indir = os.path.join(raw_dir, subdir)
        files = glob(os.path.join(indir, '*.txt'))
        for infile in files:
            years = re.findall(r'\d\d\d\d', infile)
            print(infile, subdir, years)
            if len(years) == 1:
                year = years[0]
                year = int(year)
                with open(infile) as f:
                    lines = f.readlines()
                for line in tqdm(lines):
                    skip = False
                    if len(line.strip()) == 0:
                        skip = True

                    if not skip:
                        tokens = line.strip().split()
                        # remove @@symbol
                        try:
                            assert tokens[0].startswith('@@') or tokens[0].startswith('##')
                        except AssertionError as e:
                            print(tokens)
                            raise e
                        line_id = tokens[0]
                        tokens = tokens[1:]
                        # remove HTML tags
                        html_tokens = []
                        for t_i, token in enumerate(tokens):
                            if token == '<' and t_i < len(tokens)-1:
                                html_tokens.append(token + tokens[t_i+1])
                            elif token.startswith('<'):
                                html_tokens.append(token)

                        html_counter.update(html_tokens)
                        tokens = [t for t in tokens if t not in html_tags]
                        text = ' '.join(tokens)
                        # split on the redacted tokens
                        chunks = text.split('@ @ @ @ @ @ @ @ @ @')

                        for c_i, chunk in enumerate(chunks):
                            chunk = chunk.strip()
                            chunk_lengths[len(chunk)] += 1

                            # replace duplicate #s to avoid problems with BERT
                            chunk = re.sub(r'#+', '#', chunk)
                            if '##' in chunk:
                                print(chunk)

                            # deal with hyphens
                            chunk, reps = convert_hyphens(chunk)
                            replacement_counter.update(reps)

                            outlines.append({'id': line_id + '_' + str(c_i).zfill(5),
                                                'year': int(year),
                                                'genre': genre,
                                                'text': chunk,
                                                })

        outfile = os.path.join(outdir, subdir + '.jsonlist')
        with open(outfile, 'w') as f:
            for line in outlines:
                f.write(json.dumps(line) + '\n')

        print("Output written to", outfile)
    
    print("Done")

if __name__ == '__main__':
    main()
