import os
import re
import json
from collections import Counter
from optparse import OptionParser

import numpy as np

from common.alt_spellings import get_replacements
from common.bigrams import concat_ngrams
from common.misc import get_model_name


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--const-dir', type=str, default='/data/dalc/constitution/',
                      help='Const dir: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model used for tokenization: default=%default')
    parser.add_option('--alt-spellings', action="store_true", default=False,
                      help="Do replacements for alt spellings: default=%default")
    parser.add_option('--bigrams', action="store_true", default=False,
                      help="Use bigrams: default=%default")
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed (only relevant for max-tokens): default=%default')

    (options, args) = parser.parse_args()

    const_dir = options.const_dir
    model = options.model
    alt_spellings = options.alt_spellings    
    use_bigrams = options.bigrams
    seed = options.seed
    np.random.seed(seed)

    model_name = get_model_name(model)

    replacements = get_replacements(tokenized=True)

    const_file = os.path.join(const_dir, 'tokenized_' + model_name, 'all.jsonlist')

    print("Loading constitution")
    const_counter = Counter()
    with open(const_file) as f:
        lines = f.readlines()
    
    for line in lines:
        line = json.loads(line)
        tokens = line['tokens']
        if alt_spellings:
            tokens = [replacements[t] if t in replacements else t for t in tokens]

        if use_bigrams:
            combined = concat_ngrams(tokens, sep=' ') 
        else:
            combined = tokens
        
        combined = [re.sub('##', '', t) for t in combined]

        const_counter.update(combined)
    print(len(const_counter))

    outfile = os.path.join(const_dir, 'tokenized_' + model_name, 'targets.tsv')
    with open(outfile, 'w') as f:
        for target, count in const_counter.most_common():
            if re.match('.*[a-zA-Z0-9].*', target.split('_')[0]) is not None:
                f.write(target + '\t' + str(count) + '\n')
    
    print("Output written to", outfile)


if __name__ == '__main__':
    main()
