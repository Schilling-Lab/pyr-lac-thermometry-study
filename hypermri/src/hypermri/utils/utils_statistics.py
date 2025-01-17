###########################################################################################################################################################

# Statistical analysis of datasets

###########################################################################################################################################################


# =========================================================================================================================================================
# Import necessary packages
# =========================================================================================================================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================

def bar_plot_with_p_value(
    data_array_list=[np.arange(3,8), np.arange(9,14), np.arange(14,19)],
    colors=['purple', 'orange', 'dodgerblue'],
    labels=['A', 'B', 'C'],
    figsize=(3, 5),
    x_pos=[0, 0.04, 0.08],
    ylim=[0, 25],
    yticks=[0, 5, 10, 15, 20, 25],
    ylabel='testdata',
    savepath='',
    plotname='testdata_bar_plot',
    ):
        
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x_pos, 
        np.mean(data_array_list, axis=1), 
        yerr=np.std(data_array_list, axis=1), 
        align='center', 
        color=colors,
        alpha=0.8, 
        ecolor='black', 
        capsize=5, 
        edgecolor='black', 
        width=0.03,
        )
    
    for i, data_array in enumerate(data_array_list):

        x = np.ones(len(data_array)) * x_pos[i]  + np.random.normal(loc=0.0, scale=0.005, size=len(data_array))
        y = data_array       
        ax.scatter(x, y, zorder=2, facecolor='k', edgecolor='k', linewidth=1, s=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90)
    
    significant_combinations = []
    combinations = [(x, y) for x in range(len(data_array_list)) for y in range(x+1, len(data_array_list))]

    for c in combinations:
        data1 = data_array_list[c[0]]
        data2 = data_array_list[c[1]]

        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    bottom, top = ylim[0], np.max(data_array_list)
    yrange = top - bottom

    for i, significant_combination in enumerate(significant_combinations):

        x1 = x_pos[significant_combination[0][0]]
        x2 = x_pos[significant_combination[0][1]]

        level = len(significant_combinations) - i

        bar_height = (yrange * 0.09 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')

        p = significant_combination[1]
        print(p)
        
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')
        
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    
    plt.savefig(plotname + '.png')
    plt.savefig(plotname + '.svg')

    plt.tight_layout()


def box_plot_with_p_value(
    data_array_list=[np.arange(3,8), np.arange(9,14), np.arange(14,19)],
    colors=['purple', 'orange', 'dodgerblue'],
    labels=['A', 'B', 'C'],
    figsize=(3, 5),
    x_pos=[0, 0.04, 0.08],
    xlim=[-0.02, 0.10],
    ylim=[0, 25],
    yticks=[0, 5, 10, 15, 20, 25],
    ylabel='testdata',
    savepath='',
    plotname='testdata_box_plot',
    ):
        
    fig, ax = plt.subplots(figsize=figsize)
    
    bplot = ax.boxplot(
                    data_array_list, 
                    vert=True, 
                    patch_artist=True, 
                    labels=labels, 
                    zorder=1, 
                    widths=0.03,
                    positions=x_pos,
                    )  

    for median in bplot['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    for i, data_array in enumerate(data_array_list):

        x = np.ones(len(data_array)) * x_pos[i]  + np.random.normal(loc=0.0, scale=0.005, size=len(data_array))
        y = data_array       
        ax.scatter(x, y, zorder=2, facecolor='k', edgecolor='k', linewidth=1, s=20)
    
    ax.set_xlim(xlim)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90)
    
    significant_combinations = []
    combinations = [(x, y) for x in range(len(data_array_list)) for y in range(x+1, len(data_array_list))]

    for c in combinations:
        data1 = data_array_list[c[0]]
        data2 = data_array_list[c[1]]

        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    bottom, top = ylim[0], np.max(data_array_list)
    yrange = top - bottom

    for i, significant_combination in enumerate(significant_combinations):

        x1 = x_pos[significant_combination[0][0]]
        x2 = x_pos[significant_combination[0][1]]

        level = len(significant_combinations) - i

        bar_height = (yrange * 0.09 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')

        p = significant_combination[1]
        print(p)
        
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')
        
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    
    plt.savefig(plotname + '.png')
    plt.savefig(plotname + '.svg')

    plt.tight_layout()