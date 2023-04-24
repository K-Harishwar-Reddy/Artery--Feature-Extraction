import numpy as np
from matplotlib import pyplot as plt

def plot_raw_filter(y_raw, y_filter, y_label):
    y_raw_signal = [y if y!=-1 else None for y in y_raw]
    idx_none = [i for i in range(len(y_raw)) if y_raw[i] == -1]
    
    fig, ax1 = plt.subplots(figsize=(8,4))
    color='navy'
    t = range(360)
    ax1.set_xlabel('Angle', fontsize="15")
    ax1.set_ylabel(y_label, color=color, fontsize="15")
    ax1.plot(t, y_raw_signal, color=color, label="Raw")
    for i, x in enumerate(idx_none):
        if i == 0:
            ax1.axvline(x=x,  linestyle="--", color="red", label="Missing")
        else:
            ax1.axvline(x=x,  linestyle="--", color="red")

    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2 = ax1.twinx()
    color = 'darkorange'
    ax2.plot(t, y_filter, color=color, label="Filtered")
    
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=2, fontsize=12)

    plt.show()

    
def plot_hist_w_two_list(thickness_outer, thickness_inner, xlabel, path_to_svae):
    thickness_outer = [x for x in thickness_outer if x>=0]
    thickness_inner = [x for x in thickness_inner if x>=0]
    max_val = np.max(thickness_outer+thickness_inner)
    max_val = 2
    bins = np.linspace(0, max_val, 100)
    plt.figure(figsize=(10, 5)) 
    weights_outer = np.ones_like(thickness_outer)/float(len(thickness_outer))
    plt.hist(thickness_outer, bins=bins, alpha=0.5, weights=weights_outer, label="Media") 
    weights_inner = np.ones_like(thickness_inner)/float(len(thickness_inner))
    plt.hist(thickness_inner, bins=bins, alpha=0.5, weights=weights_inner, label="Intima")
    plt.xticks(np.arange(0, max_val+0.1, step=0.2), fontsize=15)
    plt.yticks(np.arange(0, 0.31, step=0.05), fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel("Probability", fontsize=20)
    plt.legend(loc='upper right', ncol=1, fontsize=20)
    if path_to_svae:
        plt.savefig(path_to_svae)
    plt.show()