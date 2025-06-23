import numpy as np
import pandas as pd
import altair as alt

import altair as alt
import matplotlib.pyplot as plt

# Note that the random seed for the figure in the paper was not properly recorded
# Because of sampling, different seeds, will produce slightly different plots
np.random.seed(24)

print("Loading data")
corpora = ['hein', 'statutes', 'elliots', 'evans', 'farrands', 'founders']

dfs = []
for corpus in corpora:
    infile = 'plotting/plot_data/char_lm_data/' + corpus + '_ppl.tsv'
    df = pd.read_csv(infile, sep='\t', header=0)
    df['corpus'] = corpus
    if len(df) > 5000:
        df = df.sample(5000)
    dfs.append(df)
    print(corpus, len(df), df['score'].mean(), np.std(df['score'].values), np.mean(df['score'] > 0.995))
    
df = pd.concat(dfs)

# Plot the PDF version

fig, ax = plt.subplots(figsize=(4, 4))

for c_i, corpus in enumerate(corpora):
    subset = df[df['corpus'] == corpus]
    print(corpus, len(subset))
    scores = sorted(subset['score'])

    yoffset = 0
    if corpus == 'founders':
        yoffset = -0.06
    if corpus == 'statutes':
        yoffset = +0.03
    if corpus == 'farrands':
        yoffset = +0.03
    
    ax.plot((np.arange(len(scores)))/len(scores) + 1/5000, scores)
    ax.scatter((np.arange(len(scores)))/len(scores) + 1/5000, scores, s=2, rasterized=True)
    if corpus == 'hein':
        ax.text(0.34, -0.71, corpus.title(), c='C' + str(c_i))
    else:
        ax.text(((np.arange(len(scores)))/len(scores) + 1/5000)[0] - 0.00017, scores[0] + yoffset, corpus.title(), c='C' + str(c_i))

ax.set_xscale('log')
ax.set_xlim(10e-6, 1)

ax.set_xlabel('Relative document rank (logged)')
ax.set_ylabel('Document perplexity')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('plots/kenlm_qa.pdf', bbox_inches='tight')
print("Plot saved to plots/kenlm_qa.pdf")

# Convert the data to a format suitable for Altair plotting

corpus_vector = []
rank_vector = []
scores_vector = []
doc_vector = []

for c_i, corpus in enumerate(corpora):    
    subset = df[df['corpus'] == corpus]
    scores = subset['score'].values
    names = subset['id'].values
    order = np.argsort(scores)
    scores = [scores[i] for i in order]
    names = [names[i].title() for i in order]

    corpus_vector.extend([corpus.title()] * len(scores))
    scores_vector.extend(np.array(scores))
    rank_vector.extend(np.arange(len(scores)) / len(scores) + 1/5000)
    doc_vector.extend(names)

df_plot = pd.DataFrame({'Source': corpus_vector, 'Rank': rank_vector, 'Score': scores_vector, 'Document': doc_vector})

domain = ['Hein', 'Statutes', 'Elliots', 'Evans', 'Farrands', 'Founders']
range_ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']#, #e377c2]


lines = alt.Chart(df_plot).mark_line().encode(
    x=alt.X('Rank', scale=alt.Scale(type='log'), axis=alt.Axis(title='Relative document rank (logged)')),
    y=alt.Y('Score', scale=alt.Scale(domain=[-1.45, -0.48]), axis=alt.Axis(title='Proportion of words in dictionary')),
    color=alt.Color('Source').scale(domain=domain, range=range_)
)

points = alt.Chart(df_plot).mark_point(filled=True, size=20).encode(
    x=alt.X('Rank'),
    y='Score',
    color='Source',
    tooltip=['Source', 'Score'],
)

chart = (lines + points).interactive().configure_axis(
    grid=False,
)

chart.save('plots/kenlm_qa.html')
print('Plot saved to plots/kenlm_qa.html')
