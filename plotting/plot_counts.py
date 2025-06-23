import json

import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

# Plot PDF version
with open('plotting/plot_data/tokens_by_year_by_source.json') as f:
    tokens_by_year_by_source = json.load(f)

sources = sorted(tokens_by_year_by_source)

fig, axes = plt.subplots(nrows=6, figsize=(5, 5), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.4)

axis = 0
for i, source in enumerate(sources):
    if source.startswith('Founders:'):
        pass
    elif source.startswith('The P'):
        pass
    else:
        name = source
        years = sorted([int(y) for y in tokens_by_year_by_source[source]])
        counts = [tokens_by_year_by_source[source][str(y)] for y in years]
        year_chunks = [[years[0]]]
        count_chunks = [[counts[0]]]
        for year_i, year in enumerate(years[1:]):
            if year == year_chunks[-1][-1] + 1:
                year_chunks[-1].append(year)
                count_chunks[-1].append(counts[year_i+1])
            else:
                year_chunks.append([year])
                count_chunks.append([counts[year_i+1]])
        for chunk_i, year_chunk in enumerate(year_chunks):
            count_chunk = count_chunks[chunk_i]        
            if chunk_i == 0:
                axes[axis].plot(year_chunk, count_chunk, 'C0')
            else:
                axes[axis].plot(year_chunk, count_chunk, 'C0')
        axes[axis].text(1637, 6e6, name[0].upper() + name[1:])
        if source == 'elliots' or source == "Farrand's Records":
            s = 14
        else:
            s = 5
        axes[axis].scatter(years, counts, s=s)
        #axes[axis].legend(loc='upper left')
        #axes[i].set_ylabel('')
        axes[axis].fill_between([1760, 1800], [-5e5, -5e5], [1.05e7, 1.05e7], color='k', alpha=0.05)
        axes[axis].plot([1787, 1787], [-5e5, 1.05e7], c='k', linestyle='dotted', alpha=0.5)
        axes[axis].set_ylim([-5e5, 1.05e7])
        axis += 1
    
plt.savefig('plotting/plots/subset_counts.pdf', bbox_inches='tight')
print("Plot saved to plotting/plots/subset_counts.pdf")

# Plot Altair version
df_counts = pd.read_csv('plotting/plot_data/tokens_by_year_by_source.csv', header=0)

sources = sorted(set(df_counts['Source'].values))
charts = []

df_dashed = pd.DataFrame({'x': [1787, 1787]})
df_rect = pd.DataFrame({'x': [1760, 1800], 'y': [0, 1e7]})

min_year = min(df_counts['Year'].values)
max_year = max(df_counts['Year'].values)

ticks = False
for source_i, source in enumerate(sources):
    subset = df_counts[df_counts['Source'] == source]
    
    if source_i == 5:
        ticks = True
    
    lines = alt.Chart(subset).mark_line().encode(
        x=alt.X('Year:Q', scale=alt.Scale(domain=[min_year, max_year]), axis=alt.Axis(format='i', labels=ticks), title=None),
        y=alt.Y('Count', scale=alt.Scale(domain=[-5e5, 1.05e7]), axis=alt.Axis(title=None)),
        detail='Detail:N'
    )
    
    points = alt.Chart(subset).mark_point(filled=True).encode(
        x=alt.X('Year:Q'),
        y=alt.Y('Count', scale=alt.Scale(domain=[-5e5, 1.05e7])),
        tooltip=['Source', 'Year', 'Count']
    )

    dashed = alt.Chart(df_dashed).mark_rule(strokeDash=[2,2], color='black').encode(
        x='x',
    )

    rect = alt.Chart(df_rect).mark_rect(color='black').encode(
        x='min(x)',
        x2='max(x)',
        opacity=alt.value(0.05),        
    )
    
    chart = (lines+points+rect+dashed).properties(
        width=300,
        height=50,
        title=alt.TitleParams(text=source, fontSize=12, offset=-20, fontWeight=400, anchor='start', frame='group', dx=10),
    )
    charts.append(chart)

chart = alt.vconcat(charts[0], charts[1], charts[2], charts[3], charts[4], charts[5]).configure(
    concat=alt.CompositionConfig(spacing=0)
)

final = chart.interactive(bind_y=False).configure_axis(
    grid=False,
)

final.save('plotting/plots/subset_counts.html')
print("Plot saved to plotting/plots/subset_counts.html")