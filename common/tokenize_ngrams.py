import json
from optparse import OptionParser

from transformers import BertTokenizer, RobertaTokenizer

from common.bigrams import bigrams


def main():
    usage = "%prog outfile.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')

    (options, args) = parser.parse_args()

    outfile = args[0]

    model_type = options.model_type
    model_name_or_path = options.model
    tokenizer = options.tokenizer

    print("Loading model")
    if model_type == 'bert':
        tokenizer_class = BertTokenizer
    elif model_type == 'roberta':
        tokenizer_class = RobertaTokenizer
    else:
        raise ValueError("Model type not recognized")

    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    tokenized_bigrams = []
    for bigram in bigrams:        
        tokenized_terms = []
        for term in bigram:
            term_pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(term, add_special_tokens=False)]
            tokenized_terms.append(''.join(term_pieces))
        tokenized_bigrams.append(tokenized_terms)

    with open(outfile + '.bigrams', 'w') as f:
        json.dump(tokenized_bigrams, f, indent=2)

    
if __name__ == '__main__':
    main()
