import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

import altair as alt

# Load data
df = pd.read_csv('plotting/plot_data/mlm_early_vs_modern_1760-1800/jsd_scores_targets.csv')
df.sort_values(by='scaled_jsd', inplace=True, ascending=False)

with open('plotting/plot_data/token_counts_1760-1800.json') as f:
    counts = json.load(f)

df_all = pd.read_csv('plotting/plot_data/mlm_early_vs_modern_1760-1800/jsd_scores.csv')

df_all.dropna(inplace=True)

df_all['count'] = [counts[re.sub(' ', '_', t)] for t in df_all['term'].values]
df['count'] = [counts[re.sub(' ', '_', t)] for t in df['term'].values]

counts_by_term = dict(zip(df['term'].values, df['count'].values))
jsd_by_term = dict(zip(df['term'].values, df['jsd'].values))

const_terms = set(df['term'].values)
df_all['const'] = [1 if t in const_terms else 0 for t in df_all['term'].values]

df_all['intercept'] = 1
df_all['log_count'] = np.log(df_all['count'])
ols = sm.OLS(endog=df_all['jsd'], exog=df_all[['intercept', 'const', 'log_count']])
fit = ols.fit()
params = fit.params

# Plot the PDF version
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(df_all['count'], df_all['jsd'], s=6, alpha=0.4)
ax.scatter(df['count'], df['jsd'], s=6, alpha=0.8)

x = np.array([10, 1e7])
y_all = params[0] + np.log(x) * params[2]
y_const = params[0] + params[1] + np.log(x) * params[2]

ax.plot(x, y_const, c='w', linestyle='dashed')
ax.plot(x, y_all, c='w', linestyle='dashed')

ax.plot(x, y_const, c='C1', linestyle='dashed')
ax.plot(x, y_all, c='C0',  linestyle='dashed')

ax.set_xscale('log')
ax.set_xlim(10, 10**8)
ax.set_xlabel('Term count')
ax.set_ylabel('Change in meaning (JSD)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for term in ['him', 'domestic violence', 'articles', 'affirmation', 'thirty', 'three', 'abridging', 'the', 'for', 'ninth', 'seizures',  'commerce']:
    ax.text(counts_by_term[term]*1.05, jsd_by_term[term]+0.01, term)
plt.savefig('plotting/plots/early_vs_modern.pdf', bbox_inches='tight')
print("Plot saved to plotting/plots/early_vs_modern.pdf")

# Plot Altair version
alt.data_transformers.disable_max_rows()

const_terms = set(df['term'].values)
df_all['const'] = [1 if t in const_terms else 0 for t in df_all['term'].values]

subset = df_all[df_all['count'] >= 20]
subset = subset[subset['const'] == 0]

background = alt.Chart(subset).mark_point(filled=True).encode(
    x=alt.X('count:Q', scale=alt.Scale(type='log'), axis=alt.Axis(title='Term Count')),
    y=alt.Y('jsd:Q', axis=alt.Axis(title='Change in meaning (JSD)'), scale=alt.Scale(domain=[0.2, 1.0])),
    color=alt.value('#1F77B4'),
    opacity=alt.value(0.4),
    tooltip=['term:N', 'const:N']
).properties(
    width=300,
    height=200
)

subset = df_all[df_all['count'] >= 20]
subset = subset[subset['const'] == 1]

x = np.array([20, 1e7])
y_all = params.iloc[0] + np.log(x) * params.iloc[2]
y_const = params.iloc[0] + params.iloc[1] + np.log(x) * params.iloc[2]

df5 = pd.DataFrame({'x': x, 'y': y_all})
df6 = pd.DataFrame({'x': x, 'y': y_const})

line1 = alt.Chart(df5).mark_line(strokeDash=[5, 3]).encode(
    x='x',
    y='y',
    color=alt.value('#1F77B4'),
)

foreground = alt.Chart(subset).mark_point(filled=True).encode(
    x=alt.X('count:Q', scale=alt.Scale(type='log')),
    y=alt.Y('jsd:Q'),
    color=alt.value('#ff7f0e'),
    opacity=alt.value(0.8),
    tooltip=['term:N', 'const:N']
)

line2 = alt.Chart(df6).mark_line(strokeDash=[5, 3]).encode(
    x='x',
    y='y',
    color=alt.value('#ff7f0e')
)

output = (background + line1 + foreground + line2).interactive().configure_axis(
    grid=False
)

output.save('plotting/plots/change_over_time.html')
print('Plot saved to plotting/plots/change_over_time.html')

