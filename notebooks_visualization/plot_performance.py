import matplotlib
# matplotlib.use('pgf')
# matplotlib.rc('pgf', texsystem='pdflatex')  # from running latex -v
# preamble = matplotlib.rcParams.setdefault('pgf.preamble', [r'\usepackage{color}'])
# preamble.append(r'\usepackage{color}')
 
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{color}')
# # matplotlib.verbose.level = 'debug-annoying'


import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.transforms import Affine2D, ScaledTranslation

plt.rcParams.update({'font.size': 25})

df_2_class = pd.read_csv('data/2_class_scores.csv')
df_3_class = pd.read_csv('data/3_class_scores.csv')
df_exp = pd.read_csv('data/experiments.csv')

exp_cat_1 = 'Early Fusion'
exp_cat_2 = 'CNN Fusion'
exp_cat_3 = 'FC Fusion'

cmap_cont = LinearSegmentedColormap.from_list('mycmap', ['#b0cffb', '#22418e'])
cmap_disc = ListedColormap(['#b0cffb', '#7f96cf','#22418e'])
cmap_disc_rev = ListedColormap(['#7f96cf','#b0cffb'])
cmap_disc_light =  ListedColormap(['#b0cffb'])
cmap_disc_middle = ListedColormap(['#7f96cf'])
cmap_disc_dark = ListedColormap(['#22418e'])
sbcmap_1 = ListedColormap([sns.desaturate('#4c72b0', 0.75)])
sbcmap_2 = ListedColormap([sns.desaturate('#dd8452', 0.75)])
mplcmap_1 = ListedColormap(['#1f77b4'])
mplcmap_2 = ListedColormap(['#ff7f0e'])
cbcmap_1 = ListedColormap(['#0173b2'])
cbcmap_2 = ListedColormap(['#de8f05'])


def limit_err_values(df, metric: str, metric_ci: str, eps=0.001):
    y_err = np.zeros((2, len(df[metric_ci])))
    for i in range(len(df[metric_ci])):
        if df[metric][i] + df[metric_ci][i] > 1 - eps:  # eps to avoid clipping
            y_err[1, i] = 1 - df[metric][i] - eps  # eps to avoid clipping
        else:
            y_err[1, i] = df[metric_ci][i]
        if df[metric][i] - df[metric_ci][i] < 0 + eps:
            y_err[0, i] = df[metric][i] - eps  # eps to avoid clipping
        else:
            y_err[0, i] = df[metric_ci][i]
    return y_err

fig_size = (12, 6)

def plot_bar(df, legend=True, binary=True, cmap_1=cmap_disc_middle, cmap_2=cmap_disc_light):
    fig, ax = plt.subplots(figsize=fig_size)

    f1_key = 'F1-test-all-mod'
    f1_ci_key = 'F1-CI-test-all-mod'
    mcc_key = 'MCC-test-all-mod'
    mcc_ci_key = 'MCC-CI-test-all-mod'

    kwargs = {
        'kind': 'bar',
        'x': 'model',
        'ylim': (0, 1.0),
        'ax': ax,
        'width': 0.3,
        'capsize': 2,
        'ecolor': 'black',
    }

    y_err_f1 = limit_err_values(df, f1_key, f1_ci_key)
    y_err_mcc = limit_err_values(df, mcc_key, mcc_ci_key)
    
    df.plot(y=f1_key, yerr=y_err_f1, colormap=cmap_1, position=1, **kwargs)
    df.plot(y=mcc_key, yerr=y_err_mcc, colormap=cmap_2, position=0, **kwargs)

    # df.plot(kind='bar', x='model', y='F1-test', yerr='CI-F1-test', ylim=(0, 1.0), colormap=cmap_disc_middle, ax=ax, position=1, width=width, capsize=2)
    # df.plot(kind='bar', x='model', y='MCC-test', yerr='CI-MCC-test', ylim=(0, 1.0), colormap=cmap_disc_light, ax=ax, position=0, width=width)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Score')
    # plt.xlabel('Modalities')

    # Add figure title
    if binary:
        # plt.xlabel('2 Targets', fontsize=16, fontweight='bold')
        plt.xlabel('2 Targets', fontweight='bold')
    else:
        plt.xlabel('3 Targets', fontweight='bold')

    # plt.xticks(list(range(7)), [r'$\textcolor{blue}{PET}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])
    # plt.xticks(list(range(7)), [r'$\textrm{\textcolor{blue}{PET}}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])

    ax.set_position([0.1, 0.1, 0.6, 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.9), frameon=False, labels=['F1', 'MCC'])
    if not legend:
        ax.legend_.remove()
    plt.xlim(-0.55, None)

    plt.vlines([2.5, 5.5], ymin=0, ymax=1, color='black', linestyles='dashed', linewidth=3)

    # Add text for stage 1, 2 and 3
    # ax.text(0.5, 0.9, 'Stage 1', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='center')
    labels_height = 1.02
    # ax.text(1, labels_height, 'Stage 1', fontsize=14, fontweight='bold', va='bottom', ha='center')
    # ax.text(4, labels_height, 'Stage 2', fontsize=14, fontweight='bold', va='bottom', ha='center')
    # ax.text(6, labels_height, 'Stage 3', fontsize=14, fontweight='bold', va='bottom', ha='center')
    ax.text(1, labels_height, 'Stage 1', fontweight='bold', va='bottom', ha='center')
    ax.text(4, labels_height, 'Stage 2', fontweight='bold', va='bottom', ha='center')
    ax.text(6, labels_height, 'Stage 3', fontweight='bold', va='bottom', ha='center')

    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax


def plot_bar_colorcoded(df, legend=True, binary=True, cmap_1=cmap_disc_middle, cmap_2=cmap_disc_light, hatches: tuple=None, white_hatches=False, hatchwidth=1, l_black=False):
    fig, ax = plt.subplots(figsize=fig_size)

    f1_key = 'F1-test-all-mod'
    f1_ci_key = 'F1-CI-test-all-mod'
    mcc_key = 'MCC-test-all-mod'
    mcc_ci_key = 'MCC-CI-test-all-mod'

    kwargs = {
        'kind': 'bar',
        'x': 'model',
        'ylim': (0, 1.0),
        'ax': ax,
        'width': 0.3,
        'capsize': 2,
        'ecolor': 'black',
        # 'color': ['#234B04', '#8DB66B', '#E4F7D2', '#E36A5C', '#8A1A1A', '#ECB18D', '#884C7C'],
        # 'color': ['#234B04', '#8DB66B', '#C7D8B8', '#164194', '#7996D4', '#A8D0FE', '#884C7C'],
        # 'color': ['#234B04', '#8DB66B', '#C7D8B8', '#164194', '#7996D4', '#A8D0FE', '#E36A5C'],
        'color': ['#234B04', '#8DB66B', '#C7D8B8', '#164194', '#7996D4', '#A8D0FE', '#884C7C'],
    }

    y_err_f1 = limit_err_values(df, f1_key, f1_ci_key)
    y_err_mcc = limit_err_values(df, mcc_key, mcc_ci_key)
    
    if white_hatches:
        matplotlib.rcParams['hatch.linewidth'] = hatchwidth
        df.plot(y=f1_key, yerr=y_err_f1, position=1, hatch=hatches[0], edgecolor='white', linewidth=hatchwidth, **kwargs)
        df.plot(y=mcc_key, yerr=y_err_mcc, position=0, hatch=hatches[1], edgecolor='white', linewidth=hatchwidth, **kwargs)
        if l_black:
            le1 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[0])
            le2 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[1])
        else:
            le1 = matplotlib.patches.Patch(facecolor='darkgrey', edgecolor='white', linewidth=hatchwidth, hatch=hatches[0])
            le2 = matplotlib.patches.Patch(facecolor='darkgrey', edgecolor='white', linewidth=hatchwidth, hatch=hatches[1])
    else:
        df.plot(y=f1_key, yerr=y_err_f1, position=1, hatch=hatches[0], edgecolor='black', **kwargs)
        df.plot(y=mcc_key, yerr=y_err_mcc, position=0, hatch=hatches[1], edgecolor='black', **kwargs)
        le1 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[0])
        le2 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[1])

    # df.plot(kind='bar', x='model', y='F1-test', yerr='CI-F1-test', ylim=(0, 1.0), colormap=cmap_disc_middle, ax=ax, position=1, width=width, capsize=2)
    # df.plot(kind='bar', x='model', y='MCC-test', yerr='CI-MCC-test', ylim=(0, 1.0), colormap=cmap_disc_light, ax=ax, position=0, width=width)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Score', labelpad=10)
    # plt.xlabel('Modalities')

    # Add figure title
    if binary:
        # plt.xlabel('2 Targets', fontsize=16, fontweight='bold')
        plt.xlabel('2 Targets', fontweight='bold', labelpad=10)
    else:
        plt.xlabel('3 Targets', fontweight='bold', labelpad=10)

    # plt.xticks(list(range(7)), [r'$\textcolor{blue}{PET}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])
    # plt.xticks(list(range(7)), [r'$\textrm{\textcolor{blue}{PET}}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])

    ax.set_position([0.1, 0.1, 0.6, 0.8])
    ax.legend(handles=[le1, le2], loc='center left', bbox_to_anchor=(0.98, 0.75), frameon=False, labels=['F1', 'MCC'])
    if not legend:
        ax.legend_.remove()
    plt.xlim(-0.55, None)

    plt.vlines([2.5, 5.5], ymin=0, ymax=1, color='black', linestyles='dashed', linewidth=3)

    # Add text for stage 1, 2 and 3
    # ax.text(0.5, 0.9, 'Stage 1', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='center')
    labels_height = 1.02
    # ax.text(1, labels_height, 'Stage 1', fontsize=14, fontweight='bold', va='bottom', ha='center')
    # ax.text(4, labels_height, 'Stage 2', fontsize=14, fontweight='bold', va='bottom', ha='center')
    # ax.text(6, labels_height, 'Stage 3', fontsize=14, fontweight='bold', va='bottom', ha='center')
    ax.text(1, labels_height, 'Stage 1', fontweight='bold', va='bottom', ha='center')
    ax.text(4, labels_height, 'Stage 2', fontweight='bold', va='bottom', ha='center')
    ax.text(6.4, labels_height, 'Stage 3', fontweight='bold', va='bottom', ha='center')

    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax



def plot_bar_exp(df, legend=True, binary=True, cmap_1=cmap_disc_middle, cmap_2=cmap_disc_light, split=True):
    fig, ax = plt.subplots(figsize=fig_size)

    f1_key = 'F1-test'
    f1_ci_key = 'F1-CI-test'
    mcc_key = 'MCC-test'
    mcc_ci_key = 'MCC-CI-test'

    kwargs = {
        'kind': 'bar',
        'x': 'model',
        'ylim': (0, 1.0),
        'ax': ax,
        'width': 0.3,
        'capsize': 2,
        'ecolor': 'black',
    }

    y_err_f1 = limit_err_values(df, f1_key, f1_ci_key)
    y_err_mcc = limit_err_values(df, mcc_key, mcc_ci_key)
    
    df.plot(y=f1_key, yerr=y_err_f1, colormap=cmap_1, position=1, **kwargs)
    df.plot(y=mcc_key, yerr=y_err_mcc, colormap=cmap_2, position=0, **kwargs)

    # df.plot(kind='bar', x='model', y='F1-test', yerr='CI-F1-test', ylim=(0, 1.0), colormap=cmap_disc_middle, ax=ax, position=1, width=width, capsize=2)
    # df.plot(kind='bar', x='model', y='MCC-test', yerr='CI-MCC-test', ylim=(0, 1.0), colormap=cmap_disc_light, ax=ax, position=0, width=width)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Score')
    # plt.xlabel('Modalities')

    # Add figure title
    if binary:
        # plt.xlabel('2 Targets', fontsize=16, fontweight='bold')
        plt.xlabel('2 Targets', fontweight='bold')
    else:
        plt.xlabel('3 Targets', fontweight='bold')

    # plt.xticks(list(range(7)), [r'$\textcolor{blue}{PET}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])
    # plt.xticks(list(range(7)), [r'$\textrm{\textcolor{blue}{PET}}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])

    ax.set_position([0.1, 0.1, 0.6, 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.9), frameon=False, labels=['F1', 'MCC'])
    if not legend:
        ax.legend_.remove()
    plt.xlim(-0.55, None)

    if split:
        labels_height = 1.02
        plt.vlines([1.5, 3.5], ymin=0, ymax=1, color='black', linestyles='dashed', linewidth=3)
        ax.text(0.5, labels_height, exp_cat_1, fontsize=14, fontweight='bold', va='bottom', ha='center')
        ax.text(2.5, labels_height, exp_cat_2, fontsize=14, fontweight='bold', va='bottom', ha='center')
        ax.text(4, labels_height, exp_cat_3, fontsize=14, fontweight='bold', va='bottom', ha='center')

    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax



def plot_bar_exp_cc(df, legend=True, binary=True, cmap_1=cmap_disc_middle, cmap_2=cmap_disc_light, hatches: tuple=None, white_hatches=False, hatchwidth=1, split=False, l_black=False, cvec=None):
    fig, ax = plt.subplots(figsize=fig_size)

    f1_key = 'F1-test'
    f1_ci_key = 'F1-CI-test'
    mcc_key = 'MCC-test'
    mcc_ci_key = 'MCC-CI-test'

    kwargs = {
        'kind': 'bar',
        'x': 'model',
        'ylim': (0, 1.0),
        'ax': ax,
        'width': 0.3,
        'capsize': 2,
        'ecolor': 'black',
        # 'color': ['#E36A5C', '#E36A5C', '#E36A5C', '#E36A5C', '#E36A5C'],
        # 'color': ['#7f96cf', '#7f96cf', '#7f96cf', '#7f96cf', '#E36A5C'],
        # 'color': ['#164194', '#164194', '#164194', '#164194', '#E2A476'],
        # 'color': ['#B1967D', '#B1967D', '#B1967D', '#B1967D', '#164194'],
        'color': cvec,
        # 'color': ['#E6AD57', '#E6AD57', '#E6AD57', '#E6AD57', '#164194'], # yellows
        # 'color': ['#F48320', '#F48320', '#F48320', '#F48320', '#164194'], # oranges
        # 'color': ['#db853b', '#db853b', '#db853b', '#db853b', '#164194'], # sns oranges middle
        # 'color': ['#ebc9a0', '#ebc9a0', '#ebc9a0', '#ebc9a0', '#164194'], # sns oranges light
    }

    y_err_f1 = limit_err_values(df, f1_key, f1_ci_key)
    y_err_mcc = limit_err_values(df, mcc_key, mcc_ci_key)
    
    if white_hatches:
        matplotlib.rcParams['hatch.linewidth'] = hatchwidth
        df.plot(y=f1_key, yerr=y_err_f1, position=1, hatch=hatches[0], edgecolor='white', linewidth=hatchwidth, **kwargs)
        df.plot(y=mcc_key, yerr=y_err_mcc, position=0, hatch=hatches[1], edgecolor='white', linewidth=hatchwidth, **kwargs)
        if l_black:
            le1 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[0])
            le2 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[1])
        else:
            le1 = matplotlib.patches.Patch(facecolor='darkgrey', edgecolor='white', linewidth=hatchwidth, hatch=hatches[0])
            le2 = matplotlib.patches.Patch(facecolor='darkgrey', edgecolor='white', linewidth=hatchwidth, hatch=hatches[1])
    else:
        df.plot(y=f1_key, yerr=y_err_f1, position=1, hatch=hatches[0], edgecolor='black', **kwargs)
        df.plot(y=mcc_key, yerr=y_err_mcc, position=0, hatch=hatches[1], edgecolor='black', **kwargs)
        le1 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[0])
        le2 = matplotlib.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor='black', linewidth=hatchwidth, hatch=hatches[1])

    # df.plot(kind='bar', x='model', y='F1-test', yerr='CI-F1-test', ylim=(0, 1.0), colormap=cmap_disc_middle, ax=ax, position=1, width=width, capsize=2)
    # df.plot(kind='bar', x='model', y='MCC-test', yerr='CI-MCC-test', ylim=(0, 1.0), colormap=cmap_disc_light, ax=ax, position=0, width=width)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Score', labelpad=10)
    # plt.xlabel('Modalities')

    # Add figure title
    if binary:
        # plt.xlabel('2 Targets', fontsize=16, fontweight='bold')
        plt.xlabel('2 Targets', fontweight='bold', labelpad=10)
    else:
        plt.xlabel('3 Targets', fontweight='bold', labelpad=10)

    # plt.xticks(list(range(7)), [r'$\textcolor{blue}{PET}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])
    # plt.xticks(list(range(7)), [r'$\textrm{\textcolor{blue}{PET}}$', 'MRI', 'Tabular', 'PET-MRI', 'PET-Tabular', 'MRI-Tabular', 'All modalities'])

    ax.set_position([0.1, 0.1, 0.6, 0.8])
    ax.legend(handles=[le1, le2], loc='center left', bbox_to_anchor=(1.1, 0.9), frameon=False, labels=['F1', 'MCC'])
    if not legend:
        ax.legend_.remove()
    plt.xlim(-0.55, None)

    if split:
        plt.vlines([1.5, 3.5], ymin=0, ymax=1, color='black', linestyles='dashed', linewidth=3)

    # Add text for stage 1, 2 and 3
    # ax.text(0.5, 0.9, 'Stage 1', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='center')
    if split:
        labels_height = 1.02
        # ax.text(0.5, labels_height, exp_cat_1, fontsize=14, fontweight='bold', va='bottom', ha='center')
        # ax.text(2.5, labels_height, exp_cat_2, fontsize=14, fontweight='bold', va='bottom', ha='center')
        # ax.text(4, labels_height, exp_cat_3, fontsize=14, fontweight='bold', va='bottom', ha='center')

        ax.text(0.5, labels_height, exp_cat_1, fontweight='bold', va='bottom', ha='center')
        ax.text(2.5, labels_height, exp_cat_2, fontweight='bold', va='bottom', ha='center')
        ax.text(4.3, labels_height, exp_cat_3, fontweight='bold', va='bottom', ha='center')


    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == "__main__":
    # fig, ax = plot_bar(df_2_class, legend=False, binary=True)
    # fig.savefig('data/images/2_class_scores_bars.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar(df_3_class, binary=False)
    # fig.savefig('data/images/3_class_scores_bars.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar(df_2_class, legend=False, binary=True, cmap_1=sbcmap_1, cmap_2=sbcmap_2)
    # fig.savefig('data/images/2_class_scores_bars_sns.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar(df_3_class, binary=False, cmap_1=sbcmap_1, cmap_2=sbcmap_2)
    # fig.savefig('data/images/3_class_scores_bars_sns.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar(df_2_class, legend=False, binary=True, cmap_1=mplcmap_1, cmap_2=mplcmap_2)
    # fig.savefig('data/images/2_class_scores_bars_sns2.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar(df_3_class, binary=False, cmap_1=mplcmap_1, cmap_2=mplcmap_2)
    # fig.savefig('data/images/3_class_scores_bars_sns2.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar(df_2_class, legend=False, binary=True, cmap_1=cbcmap_1, cmap_2=cbcmap_2)
    # fig.savefig('data/images/2_class_scores_bars_sns3.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar(df_3_class, binary=False, cmap_1=cbcmap_1, cmap_2=cbcmap_2)
    # fig.savefig('data/images/3_class_scores_bars_sns3.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=('/','\\'))
    # fig.savefig('data/images/2_class_scores_bars_colorcoded_slash.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=('/','\\'))
    # fig.savefig('data/images/3_class_scores_bars_colorcoded_slash.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=('o','.'))
    # fig.savefig('data/images/2_class_scores_bars_colorcoded_dot.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=('o','.'))
    # fig.savefig('data/images/3_class_scores_bars_colorcoded_dot.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=('/','x'))
    # fig.savefig('data/images/2_class_scores_bars_colorcoded_x.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=('/','x'))
    # fig.savefig('data/images/3_class_scores_bars_colorcoded_x.png', dpi=300, transparent=True, bbox_inches='tight')


    # fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=(None,'/'), white_hatches=True)
    # fig.savefig('data/images/2_class_scores_bars_colorcoded_w.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=(None,'/'), white_hatches=True)
    # fig.savefig('data/images/3_class_scores_bars_colorcoded_w.png', dpi=300, transparent=True, bbox_inches='tight')


    # fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=(None,'/'), white_hatches=True, hatchwidth=2)
    # fig.savefig('data/images/2_class_scores_bars_colorcoded_wm.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=(None,'/'), white_hatches=True, hatchwidth=2)
    # fig.savefig('data/images/3_class_scores_bars_colorcoded_wm.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=(None,'/'), white_hatches=True, hatchwidth=3)
    # fig.savefig('data/images/2_class_scores_bars_colorcoded_wf.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=(None,'/'), white_hatches=True, hatchwidth=3)
    # fig.savefig('data/images/3_class_scores_bars_colorcoded_wf.png', dpi=300, transparent=True, bbox_inches='tight')

    fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=(None,'//'), white_hatches=True, hatchwidth=2)
    fig.savefig('data/images/2_class_scores_bars_colorcoded_wd.png', dpi=500, transparent=True, bbox_inches='tight')

    fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=(None,'//'), white_hatches=True, hatchwidth=2)
    fig.savefig('data/images/3_class_scores_bars_colorcoded_wd.png', dpi=500, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp(df_exp, legend=True, binary=True)
    # fig.savefig('data/images/2_class_experiments.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp(df_exp, legend=True, binary=True, cmap_1=sbcmap_1, cmap_2=sbcmap_2)
    # fig.savefig('data/images/2_class_experiments_sns.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_2_class, legend=False, binary=True, hatches=(None,'//'), white_hatches=True, hatchwidth=2, l_black=True)
    # fig.savefig('data/images/2_class_scores_bars_colorcoded_wd_lb.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_colorcoded(df_3_class, legend=True, binary=False, hatches=(None,'//'), white_hatches=True, hatchwidth=2, l_black=True)
    # fig.savefig('data/images/3_class_scores_bars_colorcoded_wd_lb.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp(df_exp, legend=True, binary=True, cmap_1=mplcmap_1, cmap_2=mplcmap_2)
    # fig.savefig('data/images/2_class_experiments_sns2.png', dpi=300, transparent=True, bbox_inches='tight')
    
    # fig, ax = plot_bar_exp(df_exp, legend=True, binary=True, cmap_1=cbcmap_1, cmap_2=cbcmap_2)
    # fig.savefig('data/images/2_class_experiments_sns3.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp_cc(df_exp, legend=True, binary=True, hatches=('/','\\'))
    # fig.savefig('data/images/2_class_experiments_hatch1.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp_cc(df_exp, legend=True, binary=True, hatches=('o','.'))
    # fig.savefig('data/images/2_class_experiments_hatch2.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp_cc(df_exp, legend=True, binary=True, hatches=('/','x'))
    # fig.savefig('data/images/2_class_experiments_hatch3.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp_cc(df_exp, legend=True, binary=True, hatches=(None,'//'), white_hatches=True, hatchwidth=2)
    # fig.savefig('data/images/2_class_experiments_hatch4.png', dpi=300, transparent=True, bbox_inches='tight')

    # fig, ax = plot_bar_exp(df_exp, legend=True, binary=True, cmap_1=sbcmap_1, cmap_2=sbcmap_2, split=True)
    # fig.savefig('data/images/2_class_experiments_sns_split.png', dpi=300, transparent=True, bbox_inches='tight')
    
    # fig, ax = plot_bar_exp(df_exp, legend=True, binary=True, cmap_1=mplcmap_1, cmap_2=mplcmap_2, split=True)
    # fig.savefig('data/images/2_class_experiments_sns2_split.png', dpi=300, transparent=True, bbox_inches='tight')
    
    # fig, ax = plot_bar_exp(df_exp, legend=True, binary=True, cmap_1=cbcmap_1, cmap_2=cbcmap_2, split=True)
    # fig.savefig('data/images/2_class_experiments_sns3_split.png', dpi=300, transparent=True, bbox_inches='tight')

    color1 = ['#EED38E', '#EED38E', '#EED38E', '#EED38E', '#164194'] # yellows middle
    color2 = ['#E6AD57', '#E6AD57', '#E6AD57', '#E6AD57', '#164194'] # yellows dark
    color3 = ['#ebc9a0', '#ebc9a0', '#ebc9a0', '#ebc9a0', '#164194'] # sns oranges light
    color4 = ['#db853b', '#db853b', '#db853b', '#db853b', '#164194'] # sns oranges middle

    for i, c in enumerate([color1, color2, color3, color4]):
        fig, ax = plot_bar_exp_cc(df_exp, legend=True, binary=True, hatches=(None,'//'), white_hatches=True, hatchwidth=2, split=True, cvec=c)
        fig.savefig(f'data/images/2_class_experiments_hatch_split{i}.png', dpi=500, transparent=True, bbox_inches='tight')