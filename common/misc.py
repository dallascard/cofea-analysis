import os
import re
from collections import Counter

def get_model_name(model_name_or_path):
    if len(os.path.split(model_name_or_path)) > 1:
        model_path, model_name = os.path.split(model_name_or_path)
        if len(model_name) == 0:
            model_name = os.path.split(model_path)[-1]
    else:
        model_name = model_name_or_path
    return model_name

    
def convert_hyphens(text):

    # Split on whitespace
    paragraphs = text.split('\n\n')
    clean_paragraphs = []

    for p in paragraphs:
        tokens = p.split()

        # Create a function to replace prefixes followed by a hyphen with the prefix
        def find_and_replace(token, prefix):
            if re.match(r'^[\W]*' + prefix + '-' + r'[\w]+', token, re.IGNORECASE):
                token = re.sub(prefix + '-', prefix, token, re.IGNORECASE)
            return token

        # Replace hyphens for particular prefixes and common words
        output_tokens = []
        replacement_counter = Counter()
        for token in tokens:
            if Counter(token)['-'] == 1:
                orig_token = token[:]
                # Remove hyphens from common prefixes
                token = find_and_replace(token, 're')
                token = find_and_replace(token, 'pre')
                token = find_and_replace(token, 'non')
                token = find_and_replace(token, 'co')
                token = find_and_replace(token, 'fore')
                token = find_and_replace(token, 'inter')
                token = find_and_replace(token, 'un')

                if orig_token != token:
                    replacement_counter[orig_token] += 1

            output_tokens.append(token)
        
        # Rejoin the tokens with spaces
        paragraph = ' '.join(output_tokens)

        # Finally, replace all remaining hyphens with spaces        
        paragraph = re.sub('-', ' ', paragraph)
        clean_paragraphs.append(paragraph)

    # Rejoin the cleaned paragraphs
    text = '\n\n'.join(clean_paragraphs)
    
    return text.strip(), replacement_counter


