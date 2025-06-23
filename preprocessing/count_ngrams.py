import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm

from common.alt_spellings import get_replacements


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--sources', type=str, default=None,
                      help='Comma-separated list of sources to use [founders|statutes|farrands|elliots|hein|evans|news] (None=all): default=%default')
    parser.add_option('--first-year', type=int, default=None,
                      help='Last year to use: default=%default')
    parser.add_option('--last-year', type=int, default=None,
                      help='Last year to use: default=%default')
    parser.add_option('--ignore-year', action="store_true", default=False,
                      help='Ignore year: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model_name_or_path = options.model
    sources = options.sources
    first = options.first_year
    last = options.last_year
    ignore_year = options.ignore_year

    if first is None:
        first = 1500

    if last is None:
        last = 1900

    replacements = get_replacements()

    outdir = os.path.join(basedir, 'ngram_counts')
    if sources is not None:
        outdir += '_' + sources
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    tokenized_dir = os.path.join(basedir, 'tokenized_' + model_name_or_path)
    parsed_dir = os.path.join(basedir, 'parsed')

    if sources is None:
        sources = ['founders', 'statutes', 'farrands', 'elliots', 'hein', 'evans', 'news']
    else:
        sources = sources.split(',')

    tag_counter = Counter()
    token_counter = Counter()
    token_tag_counters = defaultdict(Counter)
    bigram_counter = Counter()
    bigram_tag_counters = defaultdict(Counter)
    trigram_counter = Counter()
    trigram_tag_counters = defaultdict(Counter)

    with open(os.path.join(outdir, 'log.txt'), 'w') as f:
        f.write('Sources: ' + ', '.join(sources) + '\n')
        f.write('First decade: ' + str(first) + '\n')
        f.write('Last decade: ' + str(last) + '\n')

    for source in sources:
        speech_ids_excluded = Counter()
        print(source)
        infile = os.path.join(tokenized_dir, source + '.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        valid_doc_years = {line['id']: line['year'] for line in lines}        

        infile = os.path.join(parsed_dir, source + '.jsonlist')
        with open(infile) as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            line_id = line['id']
            speech_id = '_'.join(line_id.split('_')[:-1])
            valid = False
            if ignore_year:
                valid = True
            elif speech_id in valid_doc_years and first <= valid_doc_years[speech_id] <= last: 
                valid = True
            if valid:            
                sents = line['tokens']
                tags = line['tags']
                for s_i, sent in enumerate(sents):
                    tokens = [t.lower() for t in sent]
                    # use standardized spellings for constitutional terms
                    tokens = [replacements[t] if t in replacements else t for t in tokens]
                    for t_i, token in enumerate(tokens):
                        tag = tags[s_i][t_i]
                        tag_counter[tag] += 1
                        token_tag_counters[token][tag] += 1
                        token_counter[token] += 1
                        if t_i > 0:
                            bigram = tokens[t_i-1] + ' ' + token
                            bigram_counter[bigram] += 1
                            bigram_tag_counters[bigram][tags[s_i][t_i-1] + ' ' + tags[s_i][t_i]] += 1
                        if t_i > 1:
                            trigram = tokens[t_i-2] + ' ' + tokens[t_i-1] + ' ' + token
                            trigram_counter[trigram] += 1
                            trigram_tag_counters[trigram][tags[s_i][t_i-2] + ' ' + tags[s_i][t_i-1] + ' ' + tags[s_i][t_i]] += 1
            else:
                speech_ids_excluded[speech_id] += 1

        print("Excluded due to year or lack of year:", len(speech_ids_excluded))

    print("Saving tag_counts.txt")  
    with open(os.path.join(outdir, 'tag_counts.txt'), 'w') as f:
        for tag, count in tqdm(tag_counter.most_common()):
            f.write(str(tag) + '\t' + str(count) + '\n')

    print("Saving token_counts.txt")
    with open(os.path.join(outdir, 'token_counts.txt'), 'w') as f:
        for token, count in tqdm(token_counter.most_common()):
            f.write(str(token) + '\t' + str(count) + '\n')

    print("Saving bigram_counts.txt")
    with open(os.path.join(outdir, 'bigram_counts.txt'), 'w') as f:
        for bigram, count in tqdm(bigram_counter.most_common()):
            f.write(str(bigram) + '\t' + str(count) + '\n')

    print("Saving trigram_counts.txt")
    with open(os.path.join(outdir, 'trigram_counts.txt'), 'w') as f:
        for trigram, count in tqdm(trigram_counter.items()):
            f.write(str(trigram) + '\t' + str(count) + '\n')

    print("Saving token_tag_counts.json")
    with open(os.path.join(outdir, 'token_tag_counts.json'), 'w') as f:
        json.dump(token_tag_counters, f, indent=2)

    print("Saving bigram_tag_counts.json")
    bigram_top_tags = {}
    for bigram, counter in tqdm(bigram_tag_counters.items()):
        for tags, count in counter.most_common(n=1):
            bigram_top_tags[bigram] = str(tags)

    with open(os.path.join(outdir, 'bigram_tag_counts.json'), 'w') as f:
        json.dump(bigram_top_tags, f, indent=2)

    print("Saving trigram_tag_counts.json")
    trigram_top_tags = {}
    for trigram, counter in tqdm(trigram_tag_counters.items()):
        for tags, count in counter.most_common(n=1):
            trigram_top_tags[trigram] = str(tags)

    with open(os.path.join(outdir, 'trigram_tag_counts.json'), 'w') as f:
        json.dump(trigram_top_tags, f, indent=2)

if __name__ == '__main__':
    main()
