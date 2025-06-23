import os
import re
import json
from glob import glob
from optparse import OptionParser
from collections import Counter

from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from bs4 import BeautifulSoup

from common.misc import convert_hyphens


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--aa-dir', type=str, default='/data/dalc/accessible_archives/',
                      help='Base directory: default=%default')
    
    (options, args) = parser.parse_args()

    basedir = options.aa_dir    

    outdir = os.path.join(basedir, 'clean')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    articles = []
    articles.extend(process_dir(basedir, 'THEPENNSYLVANIAGAZETTE'))

    outfile = os.path.join(outdir, 'news.jsonlist')
    with open(outfile, 'w') as fo:
        for article in articles:
            fo.write(json.dumps(article) + '\n')

    print("Output written to", outfile)


def process_dir(basedir, source):

    indir = os.path.join(basedir, source)

    temp_dir = os.path.join(basedir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    files = sorted(glob(os.path.join(indir, '*.xml')))

    articles = []
    for infile in tqdm(files):
        basename = os.path.basename(infile)

        # Create a clean version of this file by converting things to unicode
        with open(infile, 'r') as f:
            text = f.read()
        text = re.sub('&deg;', '°', text)
        text = re.sub('&pound;', '£', text)
        text = re.sub('&frac12;', '½', text)
        text = re.sub('&frac14;', '¼', text)
        text = re.sub('&AElig;', 'Æ', text)
        text = re.sub('&nbsp;', ' ', text)

        # Save a clean version to a temp directory
        clean_file = os.path.join(temp_dir, basename)
        with open(clean_file, 'w') as f:
            f.write(text)

        # Read the cleaned XML
        try:
            tree = ET.parse(clean_file)
        except ParseError as e:
            print("ParseError on", infile)
            print(e)
            break

        root = tree.getroot()
        
        article = {'id': basename, 'source': 'news', 'collection': source, 'text': []}
        for child in root:            
            for c2 in child:
                if c2.tag == 'article-title':
                    cleantext = BeautifulSoup(ET.tostring(c2), "lxml").text 
                    article['title'] = cleantext
                elif c2.tag == 'publication-date':
                    cleantext = BeautifulSoup(ET.tostring(c2), "lxml").text 
                    parts = cleantext.split()
                    year = parts[-1]
                    if ',' in year:
                        year = year.split(',')[-1]
                    article['year'] = int(year)
                elif child.tag == 'body':
                    cleantext = BeautifulSoup(ET.tostring(c2), "lxml").text 
                    cleantext = re.sub(r'\s', ' ', cleantext)
                    article['text'].append(cleantext.strip())
        article['text'] = [line for line in article['text'] if len(line) > 0]
        article['text'] = '\n\n'.join(article['text'])
        
        # Replace hyphens to be consistent with COFEA
        article['text'], _ = convert_hyphens(article['text'])
        articles.append(article)

    field_counter = Counter()
    for article in articles:
        field_counter.update(article.keys())

    return articles


if __name__ == '__main__':
    main()


