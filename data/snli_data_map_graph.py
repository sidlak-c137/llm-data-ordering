import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set(style='whitegrid', font_scale=1.6, context='paper')

dataframe = pd.read_json("snli_data_map_coordinates.jsonl", orient="records", lines=True)
print(dataframe.columns)

# Make graph
dataframe = dataframe.sample(n=25000 if dataframe.shape[0] > 25000 else len(dataframe))

# Normalize correctness to a value between 0 and 1.
dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

main_metric = 'variability'
other_metric = 'confidence'

hue = "correct."
num_hues = len(dataframe[hue].unique().tolist())
style = "correct." if num_hues < 8 else None

fig = plt.figure(figsize=(16, 10), )
gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])

ax0 = fig.add_subplot(gs[0, :])


### Make the scatterplot.

# Choose a palette.
pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

plot = sns.scatterplot(x=main_metric,
                        y=other_metric,
                        ax=ax0,
                        data=dataframe,
                        hue=hue,
                        palette=pal,
                        style=style,
                        s=30)

# Annotate Regions.
bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
an1 = ax0.annotate("ambiguous", xy=(0.9, 0.5), xycoords="axes fraction", fontsize=15, color='black',
                va="center", ha="center", bbox=bb('black'))
an2 = ax0.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction", fontsize=15, color='black',
                va="center", ha="center", bbox=bb('r'))
an3 = ax0.annotate("hard-to-learn", xy=(0.35, 0.25), xycoords="axes fraction", fontsize=15, color='black',
                va="center", ha="center", bbox=bb('b'))


plot.legend(fancybox=True, shadow=True,  ncol=1)
plot.set_xlabel('variability')
plot.set_ylabel('confidence')

plot.set_title(f"SNLI Data Map", fontsize=17)

# Make the histograms.
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[1, 2])

plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
plott0[0].set_title('')
plott0[0].set_xlabel('confidence')
plott0[0].set_ylabel('density')

plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
plott1[0].set_title('')
plott1[0].set_xlabel('variability')

plot2 = sns.countplot(x="correct.", data=dataframe, color='#86bf91', ax=ax3)
ax3.xaxis.grid(True) # Show the vertical gridlines

plot2.set_title('')
plot2.set_xlabel('correctness')
plot2.set_ylabel('')

fig.tight_layout()
fig.savefig('../figures/data/snli_datamap.png', dpi=300)
