import os
import re
import json
from glob import glob
from optparse import OptionParser
from collections import Counter

import fasttext
from tqdm import tqdm

from common.misc import convert_hyphens


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/u/scr/nlp/data/COFEA',
                      help='Base directory: default=%default')
    parser.add_option('--fasttext', type=str, default='lid.176.bin',
                      help='Location of fasttext language id model: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir    
    indir = os.path.join(basedir, 'orig')
    outdir = os.path.join(basedir, 'clean')
    fasttext_model_file = options.fasttext
    lid_model = fasttext.load_model(fasttext_model_file)    

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    files = sorted(glob(os.path.join(indir, '*.json')))

    replacement_counter = Counter()

    for infile in files:

        n_with_year = 0
        n_with_decade = 0
        n_with_year_and_decade = 0
        n_without_year_or_decade = 0
        n_mismatch = 0
        lang_counter = Counter()
        printed = 0

        print(infile)
        basename = os.path.basename(infile)
        subset = basename.split('_')[0]
        docs = []
        with open(infile) as f:
            data = json.load(f)
        print("Read {:d} documents".format(len(data)))
        for d in tqdm(data):
            # drop some extra newlines
            d['title'] = d['title'].strip()
            d['body'] = d['body'].strip()
            
            # remove pipe symbols, which corrupt the data
            d['title'] = re.sub('\|', '', d['title'].strip())
            d['body'] = re.sub('\|', '', d['body'].strip())
            d['title'] = re.sub('∣', '', d['title'].strip())
            d['body'] = re.sub('∣', '', d['body'].strip())
            
            # fix one document with many problems (http://founders.archives.gov/documents/Jefferson/99-01-02-9951)
            if d['id'] == 'fndrs.jefferson.99-01-02-9951':
                d['author'] = 'Wright, Robert'
                d['year'] = 1809
                d['decade'] = 1800
                d['collection'] = 'Jefferson Papers'
            # fix another that has year and decade listed as 2000:
            elif d['id'] == 'fndrs.jefferson.01-42-02-0442-0002':
                d['year'] = 1804
                d['decade'] = 1800
            # fix one document that clearly has the wrong year/decade (17626/17606)
            elif d['id'] == 'evans.N07112':
                d['year'] = 1762
                d['decade'] = 1760
            # fix years and decades for Elliot's debates (many listed as "2018")
            elif d['source'] == "Elliot's Debates":
                if 'year' in d and int(d['year']) == 2018:
                    d.pop('year')
                d['decade'] = 1780
            # convert all years and decades to ints
            if 'year' in d:
                d['year'] = int(d['year'])
            if 'decade' in d:
                d['decade'] = int(d['decade'])
            # replace duplicate pounds signs to avoid confusing bert
            if '##' in d['body']:
                print("Replacing #+ in", d['id'])
                d['body'] = re.sub(r'#+', '#', d['body'])
                assert '##' not in d['body']

            # deal with hyphens in the body text
            d['body'], reps = convert_hyphens(d['body'])
            replacement_counter.update(reps)

            # Only keep those documents with body text, a decade, and drop Editorial notes
            if 'body' in d and len(d['body'].strip()) > 0:
                # guess the langauge using fasttext; first replace all whitespace with a space
                text = re.sub(r'\s+', ' ', d['body'])
                lang_pred = lid_model.predict(text)
                lang_conf = lang_pred[1]
                assert len(lang_conf) == 1
                d['lang'] = lang_pred[0][0]
                d['lang_conf'] = float(lang_conf[0])
                lang_counter[d['lang']] += 1

                if d['lang'] != '__label__en' and printed < 6:
                    print(d['id'])
                    print(d['body'][:200])
                    print(d['lang'])
                    print()
                    printed += 1

                if 'decade' in d and d['decade'] is not None:
                    if 'year' in d and d['year'] is not None:
                        n_with_year_and_decade += 1
                        year = int(d['year'])
                        decade = int(d['decade'])
                        if int(year//10*10) != decade:
                            n_mismatch += 1
                    else:
                        n_with_decade += 1
                elif 'year' in d and d['year'] is not None:
                    n_with_year += 1
                else:
                    n_without_year_or_decade += 1

                if subset == 'elliots':
                    d['year'] = 1787

                # exclude non-English at this stage
                if 'decade' in d and d['decade'] is not None and d['title'] != 'Editorial Note' and d['lang'] == '__label__en':                    
                    # Also exclude some Farrands documents that were not properly retrieved
                    if subset == 'farrands' and d['body'].startswith('> We were unable to find any matches for your search.'):                        
                        print("Excluding Farrands document", d['id'])
                    else:
                        docs.append(d)
                elif d['title'] != 'Editorial Note':
                    print("Excluding document", d['id'], "(editorial note)")
                elif d['lang'] == '__label__en':
                    print("Excluding document", d['id'], "(non-English)")
                else:
                    print("Excluding document", d['id'], "(no decade)")

        print("with year only:", n_with_year)
        print("with decade only:", n_with_decade)
        print("with both:", n_with_year_and_decade)
        print("with neither:", n_without_year_or_decade)
        print("both with mismatch:", n_mismatch)
        for lang, count in lang_counter.most_common(n=4):
            print(lang, count)
        print()

        with open(os.path.join(outdir, subset + '.jsonlist'), 'w') as f:
            for d in docs:
                f.write(json.dumps(d) + '\n')

    print("Done!")


if __name__ == '__main__':
    main()


