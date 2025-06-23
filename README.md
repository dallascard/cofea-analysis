# cofea-analysis

This repo contains replication code for the paper "Meaning Variation and Data Quality in the Corpus of Founding Era American English", published at ACL 2025 (see end for full citation). 

More information, along with interactive versions of all figures in the paper, are available at: https://dallascard.github.io/cofea/

### 1. Requirements

#### Data

For this work, we obtained a copy of COFEA from the original creators, and unfortunately cannot redistribute it here.

In addition, we have made use of the following data sources:
- The Pittsburgh Gazette, obtained from Accessible Archives, now History Commons.
- COCA: [https://www.english-corpora.org/coca/](https://www.english-corpora.org/coca/)
- ECCO-TCP: [https://textcreationpartnership.org/tcp-texts/ecco-tcp-eighteenth-century-collections-online/](https://textcreationpartnership.org/tcp-texts/ecco-tcp-eighteenth-century-collections-online/)
- The copy of the U.S. Constitution from the National Archives: [https://www.archives.gov/founding-docs/constitution](https://www.archives.gov/founding-docs/constitution)

#### External software:

In addition to the python packages listed below, a few additional resources are needed:
- for estimating semantic change or variation, clone the SBSCD repo: [https://github.com/dallascard/sbscd](https://github.com/dallascard/sbscd)
- for doing the OCR assessment with a character language model, use kenlm: [https://github.com/kpu/kenlm](https://github.com/kpu/kenlm)

#### Python packages used:
- accelerate
- altair
- beautifulsoup4
- datasets
- gensim
- fasttext
- lxml
- matplotlib
- numpy
- pandas
- spacy
- scipy
- smart-open
- statsmodels
- tqdm
- transformers

A `requirements.txt` has also been included with this repo.

After setting up the environment, it is also necessary to run `python -m spacy download en_core_web_sm`. (If using `uv`, first run `uv pip install pip`, and then `uv run spacy download es_core_news_md`)

It is also necessary to download the fasttext language id file (`lid.176.bin`) from: https://fasttext.cc/docs/en/language-identification.html

### 2. Replication Code

All steps needed to replicate the analyses and plots in the paper are given below, to be run in order. Parts for which intermediate outputs have already been included in this repo, or those that are only relevant to additional analyses in the Appendix, have been marked as optional. Note that most scripts assume that everything will be placed in a base directory called `/data/dalc/`. This can be overriden, but would have to be set using option flags for each script.

#### Constitution: 
- convert a .tsv file to .jsonlist: `python -m constitution.import_constitution`
- tokenize the text with BERT: `python -m constitution.tokenize_constitution`

#### COCA: 
- import the raw files: `python -m coca.import_coca`
- tokenize with BERT: `python -m coca.tokenize_coca`

#### Pennyslvania Gazette
- import The Pennsylvania Gazette files: `python -m gazettes.import_gazettes`
- tokenize TPG: `python -m gazettes.tokenize_gazettes`
- parse TPG: `python -m gazettes.parse_gazettes`

#### ECCO (optional; only used for LLM-based OCR assessment)
- import ECCO-TCP files: `python -m ecco.import_ecco`
- tokenize them: `python -m ecco.tokenize_ecco`

#### Import and preprocessing of COFEA
- import COFEA from raw files: `python -m preprocessing.import_cofea`
- parse with spacy to get POS tags (for bigrams): `python -m preprocessing.parse_cofea`
- do tokenization: `python -m preprocessing.tokenize_cofea`
- filter out tokenized documents based on length and language: `python -m preprocessing.filter_tokenized`

#### Finding alternate spellings (optional)
- export parsed to raw to train word vectors: `python -m word2vec.export_raw_text --lower --from-parsed --ignore-year`
- get vectors trained on the parsed corpora: `python -m word2vec.train_vectors`
- generate candidates: `python -m preprocessing.find_alt_spellings`
- manually filter output (results are given in `common.alt_spellings`)
- get tokenized versions: `python -m common.tokenize_alt_spellings outfile.json`
- manually put tokenized replacements into `common/alt_spellings` (already done)

#### Getting phrases (optional)
- count bigrams and trigrams (uses alt spellings from above): `python -m preprocessing.count_ngrams --ignore-year`
- use NPMI to find phrases: `python -m preprocessing.find_phrases`
- manually put results into files in `common`
- get tokenized versions: `python -m common.tokenize_ngrams` (done and added to `common/bigrams.py`)

#### MLM and variation: change over time
- export the targets for detecting semantic change: `python -m mlm.export_const_targets --bigrams --alt-spellings`
- export the text for continued pretraining: `python -m mlm.export_for_pretraining --coca-dir /data/dalc/COCA/ --alt-spellings --output-subdir mlm_pretraining_early_vs_modern`
- export the text to be embedded: `python -m mlm.export_for_early_vs_modern --pre 1801 --post 1759 --alt-spellings`
- in SBSCD (continue running MLM training): `python -m general.run_mlm --basedir /data/dalc/COFEA/ --data-dir /data/dalc/COFEA/mlm_pretraining_early_vs_modern_bert-large-uncased-val0.05_plus_coca/`
- in SBSCD (index the constitutional terms): `python -m general.index_terms --basedir /data/dalc/COFEA/ --data-dir /data/dalc/COFEA/mlm_early_vs_modern_1760-1800/ --targets-file /data/dalc/constitution/tokenized_bert-large-uncased/targets.tsv --max-terms 10000 --min-count 50 --stratified`
- in SBSCD (get substitutes): `python -m general.get_substitutes --basedir /data/dalc/COFEA/ --infile /data/dalc/COFEA/mlm_early_vs_modern_1760-1800/all.jsonlist --trained-model-dir /data/dalc/COFEA/mlm_pretraining_early_vs_modern_bert-large-uncased-val0.05_plus_coca/model/ --top-k 11`
-->
- in SBSCD (compute JSDs): `python -m general.compute_jsds --basedir /data/dalc/COFEA/ --infile /data/dalc/COFEA/mlm_early_vs_modern_1760-1800/all.jsonlist --top-k 10 --targets-file /data/dalc/constitution/tokenized_bert-large-uncased/targets.tsv`
- in SBSCD (gather top replacement terms): `python -m general.gather_top_replacements --basedir /data/dalc/COFEA --infile /data/dalc/COFEA/mlm_early_vs_modern_1760-1800/all.jsonlist --top-k 10`

#### MLM and variation: variation across sources
- export the text for continued pretraining (COFEA only): `python -m mlm.export_for_pretraining --alt-spellings --output-subdir mlm_pretraining_legal_vs_popular`
- export text to be indexed: `python -m mlm.export_for_legal_vs_popular  --pre 1801 --post 1759 --alt-spellings`
- in SBSCD (continue running MLM training): `python -m general.run_mlm --basedir /data/dalc/COFEA/ --data-dir /data/dalc/COFEA/mlm_pretraining_legal_vs_popular_bert-large-uncased-val0.05/`
- in SBSCD (index the constitutional terms and random others): `python -m general.index_terms --basedir /data/dalc/COFEA/ --data-dir /data/dalc/COFEA/mlm_legal_vs_popular_1760-1800/ --targets-file /data/dalc/constitution/tokenized_bert-large-uncased/targets.tsv --max-terms 10000`
- in SBSCD (get substitutes): `python -m general.get_substitutes --basedir /data/dalc/COFEA/ --infile /data/dalc/COFEA/mlm_legal_vs_popular_1760-1800/all.jsonlist --trained-model-dir /data/dalc/COFEA/mlm_pretraining_legal_vs_popular_bert-large-uncased-val0.05/model/ --top-k 11`
- in SBSCD (compute JSDs): `python -m general.compute_jsds --basedir /data/dalc/COFEA/ --infile /data/dalc/COFEA/mlm_legal_vs_popular_1760-1800/all.jsonlist --top-k 10 --targets-file /data/dalc/constitution/tokenized_bert-large-uncased/targets.tsv`
- in SBSCD (gather top replacement temrs): `python -m general.gather_top_replacements --basedir /data/dalc/COFEA --infile /data/dalc/COFEA/mlm_legal_vs_popular_1760-1800/all.jsonlist --top-k 10`

#### Check occurrences in constitution
- replace terms with alt spellings: `python -m constitution.normalize_spelling`
- in SBSCD (index all terms in the constitution): `python -m general.index_terms --basedir /data/dalc/constitution/ --data-dir /data/dalc/constitution/mlm_legal_vs_popular/ --targets-file /data/dalc/constitution/tokenized_bert-large-uncased/targets.tsv --min-count -1 --min-count-per-corpus -1`
- in SBSCD (get substitutes): `python -m general.get_substitutes_singles --basedir /data/dalc/constitution/ --infile /data/dalc/constitution/mlm_legal_vs_popular/all.jsonlist --trained-model-dir /data/dalc/COFEA/mlm_pretraining_legal_vs_popular_bert-large-uncased-val0.05/model/ --top-k 11`

#### Dictionary-based OCR quality evaluation
- lemmatize the tokenized data: `python -m ocr_qa.lemmatize`
- check for coverage of lemmatized terms in dictionary: `python -m ocr_qa.check_dict`

#### Language model based OCR quality evaluation (optional)
- export to individual characters for using a character language model: `python -m ocr_qa.export_to_char_format`
- install kenlm using kenlm python package and vcpkg
- train an character language model using kenlm: `lmplz --text bg.txt --arpa model.arpa -o 3 --discount_fallback`
- use kenlm model to evaluate corpora: `python -m ocr_qa.get_ppls`

#### word2vec for change over time (optional)
- export COFEA text for training word vectors: `python -m word2vec.export_raw_text --pre 1801 --post 1759 --bigrams --alt-spellings --lower`
- export COCA text for training word vectors: `python -m word2vec.export_raw_text --basedir /data/dalc/COCA/ --bigrams --alt-spellings --lower --ignore-year`
- train COFEA word vectors: `python -m word2vec.train_vectors --infile /data/dalc/COFEA/word2vec/all_raw_train_1760-1800.txt --lower`
- train COCA word vectors: `python -m word2vec.train_vectors --infile /data/dalc/COCA/word2vec/all_raw_train.txt --lower`
-->
- align the two sets of vectors: `python -m word2vec.align`
- compute the similarity values: `python -m word2vec.compute_vector_sim`

#### word2vec for variation across sources (optional)
- export legal text: `python -m word2vec.export_raw_text --legal-only --pre 1801 --post 1759 --alt-spellings --bigrams --lower`
- expport other text: `python -m word2vec.export_raw_text --non-legal --pre 1801 --post 1759 --alt-spellings --bigrams --lower`
- train legal vectors: `python -m word2vec.train_vectors --infile /data/dalc/COFEA/word2vec/all_raw_train_1760-1800_legal.txt --lower `
- repeate the analgous steps from above

#### Counting (optional)
- get token counts for certain ranges (e.g., before 1787) or subsets (evans, founders, legal): `python -m counting.count_tokens` 
- get distinctive tokens per corpus: `python -m counting.log_odds --pre 1801 --post 1759`

#### Plots
- export token counts by corpus: `python -m counting.export_count_data`
- export relative counts: `python -m counting.export_relative_counts`
- plot counts over time: `python -m plotting.plot_counts`
- plot dictionary OCR QA: `python -m plotting.plot_ocr_qa_dict`
- plot relative frequencies: `python -m plotting.plot_relative_freqs`
- plot language model OCR QA: `python -m plotting.plot_ocr_qa_ppl`
- plot change versus frequency: `python -m plotting.plot_change_vs_freq`

### 3. Citation

If you find this work useful, please include the following citation:
_forthcoming_