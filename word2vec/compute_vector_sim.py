import json
from optparse import OptionParser

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine as cosine_dist, euclidean

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--model1', type=str, default='data/COFEA/word2vec/all_raw_train_1760-1800.txt.gensim',
                      help='First model file (base): default=%default')
    parser.add_option('--model2', type=str, default='data/COCA/word2vec/COCA_aligned_to_cofea.gensim',
                      help='Second model file (to align): default=%default')
    parser.add_option('--outfile', type=str, default='data/COCA/word2vec/COCA_aligned_to_cofea.json',
                      help='Outfile: default=%default')
    parser.add_option('--euclid', action="store_true", default=False,
                      help="Use Euclidean distance rather than cosine: default=%default")

    (options, args) = parser.parse_args()

    file1 = options.model1
    file2 = options.model2
    outfile = options.outfile
    use_euclid = options.euclid


    output = compute_vector_sim(file1, file2, use_euclid=use_euclid)
    with open(outfile, 'w') as fo:
        json.dump(output, fo, indent=2)


def compute_vector_sim(file1, file2, use_euclid=False):

    model1 = Word2Vec.load(file1)
    model2 = Word2Vec.load(file2)

    vocab_m1 = set(model1.wv.index_to_key)
    vocab_m2 = set(model2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = sorted(vocab_m1 & vocab_m2)

    dists = []
    for term in tqdm(common_vocab):
        if use_euclid:
            dists.append(euclidean(model1.wv[term], model2.wv[term]))
        else:
            dists.append(cosine_dist(model1.wv[term], model2.wv[term]))

    order = np.argsort(dists)
    output = {common_vocab[i]: float(dists[i]) for i in order}
    
    return output


if __name__ == '__main__':
    main()
