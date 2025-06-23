from optparse import OptionParser

import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()


    df = pd.read_csv('plotting/plot_data/rel_freqs_post-1759_pre-1801.csv', index_col=0)

    freqs = df[['Popular', 'Founders', 'Legal']].values
    terms = df['terms'].values
    n_terms = len(terms)
    projected = np.zeros([n_terms, 2])
    normal = np.array([1, 1, 1])
    norm2 = np.sum(normal)**2
    for i in range(n_terms):
        vec = freqs[i, :]
        x, y = tern_to_cart(vec)
        projected[i, 0] = x
        projected[i, 1] = y

    x = projected[:, 0]
    y = projected[:, 1]

    a_x, a_y = tern_to_cart([1, 0, 0])
    b_x, b_y = tern_to_cart([0, 1, 0])
    c_x, c_y = tern_to_cart([0, 0, 1])


    def density_estimation(m1, m2):
        xmin = min(m1)
        xmax = max(m1)
        ymin = min(m2)
        ymax = max(m2)
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]                                                     
        positions = np.vstack([X.ravel(), Y.ravel()])                                                       
        values = np.vstack([m1, m2])                                                                        
        kernel = gaussian_kde(values)                                                                 
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z

    X, Y, Z = density_estimation(x, y)


    # Plot PDF version
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot([a_x, b_x], [a_y, b_y], 'k')
    ax.plot([c_x, b_x], [c_y, b_y], 'k')
    ax.plot([c_x, a_x], [c_y, a_y], 'k')
    ax.text(a_x-0.05, a_y-0.07, 'Popular')
    ax.text(b_x-0.12, b_y-0.07, 'Founders')
    ax.text(c_x-0.05, c_y+0.04, 'Legal')

    ax.contour(X, Y, Z, levels=8, cmap="viridis", alpha=0.8)
    ax.scatter(x, y, s=8, c='k', alpha=0.2)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('plotting/plots/simplex.pdf', bbox_inches='tight')
    print('Saved to plotting/plots/simplex.pdf')

    # Plot Altair version
    hex_vals = []
    for i in range(n_terms):
        vec = freqs[i, :]
        hex_vals.append(rgb_to_hex(vec[0]*255, vec[1]*255, vec[2]*255))

    df2 = pd.DataFrame({'d1': projected[:, 0], 'd2': projected[:, 1], 'term': terms, 'color': hex_vals})

    df3 = pd.DataFrame({'x': [a_x, b_x, b_x, c_x, a_x, c_x], 'y': [a_y, b_y, b_y, c_y, a_y, c_y], 'd':[1, 1, 2, 2, 3, 3]})

    text_df = pd.DataFrame({'text': ['Legal', 'Popular', 'Founders'], 'x': [0.5, 0, 1], 'y': [0.95, -0.07, -0.07]})

    chart = alt.Chart(df2).mark_point(filled=True).encode(
        x=alt.X('d1:Q', axis=None),
        y=alt.Y('d2', axis=None),
        color=alt.Color('color').scale(None),
        tooltip=['term'],
    )

    lines = alt.Chart(df3).mark_line().encode(
        x=alt.X('x:Q', axis=None),
        y=alt.Y('y:Q', axis=None),
        detail='d',
        color=alt.value('black')
    )

    text = alt.Chart(text_df).mark_text(fontSize=15).encode(
        x='x',
        y='y',
        text='text'
    )

    chart = (lines + chart + text).configure_axis(
        grid=False
    ).configure_view(stroke=None).interactive()

    chart.save('plotting/plots/simplex.html')
    print('Saved to plotting/plots/simplex.html')


def tern_to_cart(vec):    
    x = 0.5 * (2 * vec[1] + vec[2]) / sum(vec)
    y = np.sqrt(3) / 2 * vec[2] / sum(vec)
    return x, y

def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))


if __name__ == '__main__':
    main()
