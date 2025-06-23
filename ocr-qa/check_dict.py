import os
import re
import json
from glob import glob
from optparse import OptionParser
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import spacy


# Compare document tokens against a dictionary

def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--indir', type=str, default='/data/dalc/COFEA/tokenized_bert-large-uncased-lemmatized/',
                      help='Directory with lemmatized documents: default=%default')
    parser.add_option('--dict-file', type=str, default='ocr_qa/Websters-1913-Dictionary.csv',
                      help='Dictionary file in .csv format: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Run on only a single source [evans|hein|etc]: default=%default')

    (options, args) = parser.parse_args()

    nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])

    indir = options.indir
    dict_file = options.dict_file
    source = options.source
    outdir = os.path.join('plotting', 'plot_data', 'dict_quality')

    # Load Webster's 1913 dictionary
    df = pd.read_csv(dict_file)
    dict_words = set([str(s).lower() for s in df['Word'].values])
    len(dict_words)
    # Add a word that gets missed in loading the csv
    dict_words.add('none')

    # Load country names and add them to the dictionary
    with open('names/countries.json') as f:
        countries = json.load(f)
    for country in countries:
        dict_words.update(country.lower().split())

    # Load city, state, and county names, and add them to the dictionary
    city_df = pd.read_csv('names/us_cities_states_counties.csv', sep='|', header=0)
    place_names = set()
    for city in city_df['City'].values:
        place_names.update(city.lower().split())
    for state in city_df['State full'].values:
        place_names.update(state.lower().split())
    for county in city_df['County'].values:
        if type(county) == str:
            place_names.update(county.lower().split())
    for alias in city_df['City alias'].values:
        place_names.update(alias.lower().split())

    dict_words.update(place_names)
    
    # Add biblical names to the dictionary
    with open('names/old-testament.txt') as f:
        lines = f.readlines()
    dict_words.update([line.strip().lower() for line in lines[1:]])

    # Add common first names to the dictionary
    names_df = pd.read_csv('names/baby-names.csv', header=0)
    first_names = names_df['name'].values
    first_names = set([name.lower().replace('"', '') for name in first_names])
    dict_words.update(first_names)

    # Add titles to the dictionary
    titles = {'mr', 'mrs', 'ms', 'dr', 'capt', 'esq', 'esqr', 'genl', 'messrs', 'lieut', 'hble', 'brigadier', 'captn', 'govr', 'honble', 'majr', 'regt', 'doctr', }
    dict_words.update(titles)

    punct = {',', '.', '!', '?', '"', "'", '`', '(', ')', ':', '-', '–', '─', '$', '£', '%', '&', '[', ']', ';', '—', '•', '〉', '〈', '*', '’', '/', '“', '”', '~', '⟨', '⟩', '…', '>', '<', '{', '}', '▪'}

    def measure_quality(tokens, verbose=0):    
        
        # Remove empty tokens
        tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
        
        # filter out punctuation
        tokens = [t for t in tokens if t not in punct]
        tokens = [t.lower() for t in tokens]
       
        # Count good tokens
        good_count = 0
        bad_tokens = []
        roman_numerals = []
        for t in tokens:
            if t in dict_words:
                good_count += 1
            elif t.isnumeric():
                good_count += 1
            # Skip some common abbreviations and obsolete forms, as well as important names, bible books, common french words, etc
            elif t in {'expence', 'tis', 'servt', 'thro', 'ad', 'yr', 'obedt', 'et', 'unnecessary', 'recieve', 'psal', 'inclination', 'obt', 'monie', 'recd', 'fulfil', 'ibid', 'compleat', 'waggon', 'annum', 'acct', 'negociation', 'barbado', 'algier', 'judgement', 'agreable', 'ie', 'ce', 'qr', 'ib', 'cloathe', 'creator', 'heav', 'licence', 'anno', 'cloathing', 'beleive', 'everlaste', 'pensylvania', 'controul', 'perswade', 'cloath', 'imprisonment', 'compleate', 'fide', 'cometh', 'priviledge', 'trowser', 'receiv', 'quebec', 'daye', 'dayes', 'wayes', 'alwayes', 'tryal', 'civility', 'etc', 'dy', 'brigt', 'vertue', 'linnen', 'gal', 'sr', 'wch', 'seperate', 'rejoyce', 'superintendant', 'risque', 'cargoe', 'nearer', 'threatning', 'chearfully', 'canst', 'faithfulness', 'negroe', 'sert', 'lett', 'woud', 'yrs', 'oclock', 'paine', 'fulness', 'ld', 'breeche', 'cd', 'pp', 'houshold', 'shoud', 'barbadoe', 'wharff', 'adviseable', 'believeth', 'waye', 'massachuset', 'versaille', '2dly', 'cornwallis', 'acknowledgement', 'agst', 'methink', 'falshood', 'expresly', 'junr', 'giveth', 'recompence', 'timbere', 'shd', 'preceede', 'triumphant', 'knowlege', 'disintereste', 'unhappily', 'knoweth', 'chesnut', 'goverment', 'unavoidably', 'bc', 'saml', 'matth', 'tolerably', 'souldier', 'centum', 'independance', 'exod', 'marseille', 'apostacy', 'bona', 'chearful', 'solicitation', 'independant', 'answ', 'luk', 'ezek', 'maketh', 'instalment', 'barne', 'inliste', 'speciman', 'burgoyne', 'compleatly', 'peaceably', 'desireable', 'accomodation', 'thankfulness', 'followeth', 'paiment', 'uninterrupted', 'accoutrement', 'unsettled', 'indispensible', 'unpaid', 'undersigned', 'unprofitable', 'missisippi', 'guadaloupe', 'melasse', 'obdt', 'invariably', 'unrighteousness', 'enlightened', 'vergenne', 'oblig', 'dyott', 'eustatia', 'jour', 'qrs', 'negociate', 'peice', 'une', 'martinico', 'custis', 'incidental', 'hispaniola', 'enrol', 'honours', 'affaire', 'stopt', 'cf', 'oftentime'}:
                good_count += 1
            elif t in {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jany', 'feby', 'aprl', 'augt', 'sept', 'septr', 'sepr', 'octr', 'novr', 'decr'}: 
                good_count += 1
            # Check for plurals that may have been missed in lemmatization
            elif t[-1] == 's' and t[:-1] in dict_words:
                good_count += 1
            # Check for adverbial forms that may have been missed in lemmatization
            elif t[-2:] == 'ly' and t[:-2] in dict_words:
                good_count += 1
            elif re.sub('ly', 'le', t) in dict_words:
                good_count += 1
            elif t[-4:] == 'ness' and t[:-4] in dict_words:
                good_count += 1
            elif t[-3:] == 'eth' and t[:-3] in dict_words:
                good_count += 1
            # Correct for common British alternate spellings
            elif re.sub('our', 'or', t) in dict_words:
                good_count += 1
            elif re.sub('ll', 'l', t) in dict_words:
                good_count += 1
            elif re.sub('ck', 'c', t) in dict_words:
                good_count += 1
            elif t[-2:] == 're' and t[:-2] + 'er' in dict_words:
                good_count += 1
            elif t[-2:] == 'se' and t[:-2] + 'ze' in dict_words:
                good_count += 1
            elif t[-2:] == 'ze' and t[:-2] + 'se' in dict_words:
                good_count += 1           
            # Skip tokens that look like roman numerals
            elif len(set(t) - set('clxvi')) == 0:
                good_count += 1
                roman_numerals.append(t)
            # Skip tokens that look like ordinals (e.g., 1st, 2nd, 3rd)
            elif t[-2:] == 'st' and t[:-2].isnumeric():
                good_count += 1
            elif t[-2:] == 'nd' and t[:-2].isnumeric():
                good_count += 1
            elif t[-2:] == 'rd' and t[:-2].isnumeric():
                good_count += 1
            elif t[-2:] == 'th' and t[:-2].isnumeric():
                good_count += 1
            elif t[-1:] == 'd' and t[:-1].isnumeric():
                good_count += 1
            else:
                bad_tokens.append(t)             

        return good_count / len(tokens), bad_tokens, roman_numerals
    
    if source is not None:
        files = [os.path.join(indir, source + '.jsonlist')]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    bad_token_counter = Counter()
    roman_numeral_counter = Counter()
    for infile in files:
        print(infile)
        qualities = []
        ids = []
        lengths = []
        with open(infile) as f:
            lines = f.readlines()
        for line_i, line in enumerate(tqdm(lines)):
            line = json.loads(line)
            doc_id = line['id']
            tokens = line['tokens']
            tokens = [re.sub('##', '', t) for t in tokens]
            quality, bad_tokens, roman_numerals = measure_quality(tokens)
            bad_token_counter.update(bad_tokens)
            roman_numeral_counter.update(roman_numerals)
            qualities.append(quality)
            ids.append(doc_id)
            lengths.append(len(tokens))
        
        df = pd.DataFrame({'id': ids, 'quality': qualities, 'length': lengths})
        outfile = os.path.join(outdir, os.path.basename(infile).replace('.jsonlist', '_dict_quality.tsv'))
        df.to_csv(outfile, sep='\t', index=False)
        print(np.mean(qualities))

    print("\nMost common bad tokens:")
    for token, count in bad_token_counter.most_common(n=100):
        print(token, count)

    print("Done!")
    

if __name__ == '__main__':
    main()
