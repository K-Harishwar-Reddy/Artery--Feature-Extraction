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
    
def plot_two_in_one_col(thick_media, thick_intima, y_label=True, p_idx_intima=None, p_idx_media=None, path_to_save=None):
    
    ratio = [x/y for x, y in zip(thick_intima, thick_media)]
    plt.figure(figsize=(10, 5)) 
    plt.plot([x if x>=0 else None for x in thick_media], label="Media")
    plt.plot([x if x>=0 else None for x in thick_intima], label="Intima")
    plt.plot([x if x>=0 else None for x in ratio], label="Intima-Media Ratio")
    
    if p_idx_media is not None:
        plt.scatter(p_idx_media, np.array(thick_media)[p_idx_media], marker="x", s=100)
    if p_idx_intima is not None:
        plt.scatter(p_idx_intima, np.array(thick_intima)[p_idx_intima], marker="x", s=100)
    discard_samples = 0
    start = None
    for i, x in enumerate(thick_intima):
        if x < 0 and start is None:
            start = i
        elif x >= 0 and start is not None:
            discard_samples += 1
            if discard_samples == 1:
                plt.axvspan(start, i-1, alpha=0.4, facecolor='gray', label="Discard")
            else:
                plt.axvspan(start, i-1, alpha=0.4, facecolor='gray')
            start = None
    
    # If the last chunk of -2 goes until the end of the list
    if start is not None:
        plt.axvspan(start, i, alpha=0.4, facecolor='gray')
     
    plt.xlabel("Angle", fontsize=20)
    plt.xticks(np.arange(0, 361, step=120), fontsize=15)
    if y_label:
        plt.yticks(ticks=np.arange(0, 1.3, step=0.4), fontsize=15)
#         plt.yticks(ticks=np.arange(0, 101, step=20), fontsize=15)
        plt.ylabel("Thickness", fontsize=20)
        plt.legend(fontsize=20, framealpha=0.5, loc='upper right')
    else:
        plt.yticks(ticks=np.arange(0, 1.3, step=0.4), labels=[], fontsize=20)
    if path_to_save:
        plt.savefig(path_to_save)
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