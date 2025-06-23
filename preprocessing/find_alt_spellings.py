import os
import re
import json
from optparse import OptionParser
from collections import Counter

import numpy as np
from tqdm import tqdm

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--const-dir', type=str, default='/data/dalc/constitution/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--vector-raw-file', type=str, default='/data/dalc/COFEA/word2vec/all_raw_train.txt',
                      help='Raw text file used for creating vectors: default=%default')
    parser.add_option('--vector-file', type=str, default='/data/dalc/COFEA/word2vec/all_raw_train.txt.gensim',
                      help='Output of word2vec.train: default=%default')
    parser.add_option('--min-sim', type=float, default=0.5,
                      help='Min cosine similarity required: default=%default')

    (options, args) = parser.parse_args()

    const_dir = options.const_dir
    model_name_or_path = options.model
    raw_file = options.vector_raw_file
    vector_file = options.vector_file
    min_sim = options.min_sim

    outfile = os.path.join(const_dir, 'alt_spellings.json')

    wv_model = Word2Vec.load(vector_file)
    
    with open(raw_file) as f:
        lines = f.readlines()
    token_counter = Counter()
    for line in tqdm(lines):
        token_counter.update(line.split())
    len(token_counter)    

    const_file = os.path.join(const_dir, 'tokenized_' + model_name_or_path, 'all.jsonlist')
    with open(const_file) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    const_terms = set()
    for line in lines:
        tokens = line['tokens']
        const_terms.update([re.sub('##', '', token) for token in tokens])
    len(const_terms)

    alt_spellings = {}

    for term in sorted(const_terms):
        alt_spellings[term] = []
        if term in wv_model.wv:
            most_common = wv_model.wv.most_similar(term, topn=100)
            for other, dist in most_common:
                if np.abs(len(other) - len(term)) <= 2 and dist >= min_sim:
                    edit_distance = levenshteinDistance(other, term)
                    if edit_distance <= 2:
                        alt_spellings[term].append((other, dist, token_counter[other]))
            

    print("Saving to", outfile)
    with open(outfile, 'w') as f:
        json.dump(alt_spellings, f, indent=2)



if __name__ == '__main__':
    main()
