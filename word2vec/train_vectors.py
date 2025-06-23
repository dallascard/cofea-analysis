import json
import logging
from collections import Counter
from optparse import OptionParser

import smart_open
smart_open.open = smart_open.smart_open
from gensim import utils
from gensim.models import Word2Vec


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, infile, min_length=2, max_length=22, tokenized=False, lower=False):
        self.infile = infile
        self.min_length = min_length
        self.max_length = max_length
        self.tokenized = tokenized
        self.lower = lower

    def __iter__(self):
        for line in open(self.infile):
            # assume there's one document per line, tokens separated by whitespace
            if self.tokenized:
                yield my_preprocess(line, min_len=self.min_length, max_len=self.max_length, lower=self.lower)
            else:
                yield utils.simple_preprocess(line, min_len=self.min_length, max_len=self.max_length)


def my_preprocess(doc, min_len=2, max_len=22, lower=False):
    """
    Split a doc into tokens, assuming it has already been tokenized
    :param doc: a string
    :param min_len: min token length (characters)
    :param max_len: max token length (characters)
    :param lower: convert text to lower case
    :return:
    """
    if lower:
        tokens = doc.lower().split()
    else:
        tokens = doc.split()

    tokens = [t for t in tokens if min_len <= len(t) <= max_len]

    return tokens


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='/data/dalc/COFEA/word2vec/all_raw_train.txt',
                      help='Input raw text file: default=%default')
    parser.add_option('--size', type=int, default=100,
                      help='Vector size: default=%default')
    parser.add_option('--window', type=int, default=5,
                      help='Window size: default=%default')
    parser.add_option('--min-count', type=int, default=10,
                      help='Minimum word count: default=%default')
    parser.add_option('--workers', type=int, default=4,
                      help='Number of workers: default=%default')
    parser.add_option('--epochs', type=int, default=10,
                      help='Number of epochs: default=%default')
    parser.add_option('--min-length', type=int, default=1,
                      help='Minimum token length: default=%default')
    parser.add_option('--max-length', type=int, default=25,
                      help='Maximum token length: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--lower', action="store_true", default=False,
                      help='Convert to lower case: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outfile = infile + '.gensim'
    size = options.size
    window = options.window
    min_count = options.min_count
    workers = options.workers
    epochs = options.epochs
    min_length = options.min_length
    max_length = options.max_length
    seed = options.seed
    lower = options.lower

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print("Creating corpus")
    sentences = MyCorpus(infile,
                         min_length=min_length,
                         max_length=max_length,
                         tokenized=True,
                         lower=lower)

    print("Training model")
    model = Word2Vec(sentences=sentences,
                     vector_size=size,
                     window=window,
                     min_count=min_count,
                     seed=seed,
                     workers=workers,
                     epochs=epochs,
                     compute_loss=True)

    model.save(outfile)

    print("Doing final token count")
    vocab = Counter()
    for sent_tokens in sentences:
        vocab.update(sent_tokens)

    with open(outfile + '.vocab.json', 'w') as f:
        json.dump(vocab.most_common(), f, indent=2)


if __name__ == '__main__':
    main()
