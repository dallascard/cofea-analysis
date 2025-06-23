import os
import json
from glob import glob
from collections import Counter
from optparse import OptionParser

import numpy as np


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Base directory: default=%default')
    parser.add_option('--pre', type=int, default=None,
                      help='Only count documents from before this year: default=%default')
    parser.add_option('--post', type=int, default=None,
                      help='Only count documents from after this year: default=%default')
    parser.add_option('--smoothing', type=float, default=0.1,
                      help='Smoothing for log odds: default=%default')


    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    pre_year = options.pre
    post_year = options.post
    smoothing = options.smoothing

    if pre_year is not None and post_year is not None:
        suffix = '_' + str(post_year+1) + '-' + str(pre_year-1)
    else:
        suffix = ''
   
    tokenized_dir = os.path.join(basedir, 'tokenized_' + model)

    files = sorted(glob(os.path.join(tokenized_dir, 'counts', '*_token_counts' + suffix +'.json')))

    counts_by_corpus = {}
    for infile in files:
        basename = os.path.basename(infile)
        corpus = basename.split('_')[0]
        with open(infile) as f:
            data = json.load(f)
        counts_by_corpus[corpus] = Counter(data)    

    
    for target_corpus in ['evans', 'founders', 'hein']:
        print(target_corpus)    

        target_counter = Counter(counts_by_corpus[target_corpus])
        bg_counter = Counter()
        for corpus, counter in counts_by_corpus.items():
            bg_counter.update(counter)

        bg_total = sum(list(bg_counter.values()))
        target_total = sum(list(target_counter.values()))
        vocab = sorted(set(bg_counter).union(set(target_counter)))
        vocab_index = dict(zip(vocab, range(len(vocab))))

        bg_counts = np.zeros(len(vocab))
        for token, count in bg_counter.items():
            index = vocab_index[token]
            bg_counts[index] = count

        target_counts = np.zeros(len(vocab))
        for token, count in target_counter.items():
            index = vocab_index[token]
            target_counts[index] = count

        # add smoothing
        bg_counts_smoothed = bg_counts + smoothing
        target_counts_smoothed = target_counts + smoothing

        bg_odds_ratio = bg_counts_smoothed / (bg_counts_smoothed.sum() - bg_counts_smoothed)
        target_odds_ratio = target_counts_smoothed / (target_counts_smoothed.sum() - target_counts_smoothed)

        diff = np.log(target_odds_ratio) - np.log(bg_odds_ratio)

        variance = 1 / target_counts_smoothed + 1 / bg_counts_smoothed

        score = diff / np.sqrt(variance)

        order = np.argsort(score)[::-1]
        for i in order[:30]:
            print(vocab[i], bg_counts[i], target_counts[i])

        print()


if __name__ == '__main__':
    main()
