import os
import copy
import json
from optparse import OptionParser

import numpy as np
import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec

from common.alt_spellings import get_replacements
from common.bigrams import concat_ngrams

# align to word2vec models
# adapted from https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--model1', type=str, default='/data/dalc/COFEA/word2vec/all_raw_train_1760-1800.txt.gensim',
                      help='First model file (base): default=%default')
    parser.add_option('--model2', type=str, default='/data/dalc/data/COCA/word2vec/all_raw_vectors.txt.gensim',
                      help='Second model file (to align): default=%default')
    parser.add_option('--outfile', type=str, default='/data/dalc/data/COCA/word2vec/COCA_aligned_to_cofea.gensim',
                      help='Outfile: default=%default')
    parser.add_option('--const-dir', type=str, default=None,
                      help='Constitution directory (to exclude constitutional terms from alignment): default=%default')
    parser.add_option('--tokenizer', type=str, default='bert-large-uncased',
                      help='Name of tokenizer used: default=%default')

    (options, args) = parser.parse_args()

    file1 = options.model1
    file2 = options.model2
    outfile = options.outfile
    const_dir = options.const_dir
    tokenizer = options.tokenizer

    aligned = align(file1, file2, const_dir=const_dir, tokenizer=tokenizer)
    
    aligned.save(outfile)


def align(file1, file2, const_dir=None, tokenizer='bert-large-uncased'):

    replacements = get_replacements()

    exclude = set()
    
    if const_dir is not None:
        const_file = os.path.join(const_dir, 'tokenized_' + tokenizer, 'all.jsonlist')

        print("Loading constitution")

        with open(const_file) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            tokens = line['tokens']
            tokens = [replacements[t] if t in replacements else t for t in tokens]
            combined = concat_ngrams(tokens) 
            exclude.update(combined)

    print("Excluding {:d} terms".format(len(exclude)))

    model1 = Word2Vec.load(file1)
    model2 = Word2Vec.load(file2)

    aligned = smart_procrustes_align_gensim(model1, model2, exclude=exclude)

    return aligned


def smart_procrustes_align_gensim(base_embed, other_embed, words=None, exclude=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words, exclude=exclude)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

def intersection_align_gensim(m1, m2, words=None, exclude=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    If 'exclude' is set (as a list or set), exclude these words from the intersection vocab
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    if exclude is not None:
        common_vocab = common_vocab - set(exclude)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    m1_copy = copy.deepcopy(m1)
    m2_copy = copy.deepcopy(m2)

    for m in [m1_copy, m2_copy]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1_copy, m2_copy)





if __name__ == '__main__':
    main()
