import os
import re
import json
from optparse import OptionParser
from collections import Counter

import numpy as np

from common.alt_spellings import get_replacements


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--constitution-dir', type=str, default='/data/dalc/constitution/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name: default=%default')
    parser.add_option('--subdir', type=str, default='ngram_counts',
                      help='Sub directory: default=%default')
    parser.add_option('--min-count', type=int, default=5,
                      help='Min count: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    constitution_dir = options.constitution_dir
    model_name = options.model
    subdir = options.subdir
    min_count = options.min_count

    ngram_dir = os.path.join(basedir, subdir)

    token_counter = Counter()
    bigram_counter = Counter()

    replacements = get_replacements()

    print("Loading token counts")
    with open(os.path.join(ngram_dir, 'token_counts.txt')) as f:
        for line in f:
            token, count = line.strip().split('\t')
            token_counter[token] = int(count)

    print("Loading bigram counts")
    with open(os.path.join(ngram_dir, 'bigram_counts.txt')) as f:
        for line in f:
            bigram, count = line.strip().split('\t')
            bigram_counter[bigram] = int(count)

    print("Loading bigram tag counts")
    with open(os.path.join(ngram_dir, 'bigram_tag_counts.json')) as f:
        bigram_top_tags = json.load(f)

    const_bigrams = set()
    const_trigrams = set()
    with open(os.path.join(constitution_dir, 'tokenized_' + model_name, 'all.jsonlist')) as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        tokens = line['tokens']
        tokens = [re.sub('##', '', t) for t in tokens]
        # use standardized spellings for constitutional terms
        tokens = [replacements[t] if t in replacements else t for t in tokens]
        for t_i, token in enumerate(tokens):
            if t_i > 0:
                bigram = tokens[t_i-1] + ' ' + token
                const_bigrams.add(bigram)
            if t_i > 1:
                trigram = tokens[t_i-2] + ' ' + tokens[t_i-1] + ' ' + token
                const_trigrams.add(trigram)

    for b in const_bigrams:
        print(b)

    total_tokens = sum(token_counter.values())
    total_bigrams = sum(bigram_counter.values())

    print("Making arrays")
    bigrams = sorted([b for b, c in bigram_counter.items() if c >= min_count])
    bigram_counts = np.array([bigram_counter[b] for b in bigrams])
    first_unigram_counts = np.array([token_counter[b.split()[0]] for b in bigrams])
    second_unigram_counts = np.array([token_counter[b.split()[1]] for b in bigrams])

    print("Computing PMI for bigrams")
    pmi = np.log(bigram_counts) - np.log(first_unigram_counts) - np.log(second_unigram_counts) - np.log(total_bigrams) + 2 * np.log(total_tokens)
    npmi = pmi / -(np.log(bigram_counts) - np.log(total_bigrams))

    print(len(pmi), len(bigrams))

    print("Saving data")
    order = np.argsort(pmi)[::-1]
    outfile = os.path.join(ngram_dir, 'bigram_pmi.txt')
    with open(outfile, 'w') as f:
        for i in order:
            bigram = bigrams[i]
            f.write(bigram + '\t' + bigram_top_tags[bigram] + '\t' + str(pmi[i]) + '\n')

    order = np.argsort(npmi)[::-1]
    outfile = os.path.join(ngram_dir, 'bigram_npmi.txt')
    with open(outfile, 'w') as f:
        for i in order:
            bigram = bigrams[i]
            f.write(bigram + '\t' + bigram_top_tags[bigram] + '\t' + str(npmi[i]) + '\t' + str(bigram_counter[bigram]) + '\n')

    order = np.argsort(npmi)[::-1]
    outfile = os.path.join(ngram_dir, 'const_bigram_npmi.txt')
    with open(outfile, 'w') as f:
        for i in order:
            bigram = bigrams[i]
            if bigram in const_bigrams:
                f.write(bigram + '\t' + bigram_top_tags[bigram] + '\t' + str(npmi[i]) + '\t' + str(bigram_counter[bigram]) + '\n')


if __name__ == '__main__':
    main()
