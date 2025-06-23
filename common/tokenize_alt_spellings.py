import json
from glob import glob
from collections import defaultdict
from optparse import OptionParser

from transformers import BertTokenizer, RobertaTokenizer

from common.alt_spellings import alt_spellings


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

    tokenized_alt_spellings = defaultdict(list)
    for term, replacements in alt_spellings.items():
        term_pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(term, add_special_tokens=False)]
        for replacement in replacements:
            replacement_pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(replacement, add_special_tokens=False)]
            tokenized_alt_spellings[''.join(term_pieces)].append(''.join(replacement_pieces))

    with open(outfile, 'w') as f:
        json.dump(tokenized_alt_spellings, f, indent=2)


if __name__ == '__main__':
    main()
