import numpy as np
import scipy
from scipy import stats
from matplotlib import pyplot as plt

def plot_hist_w_two_list(thickness_outer, thickness_inner, xlabel, excludes, path_to_svae):
    for val in excludes:
        thickness_outer = [x for x in thickness_outer if x!=val]
        thickness_inner = [x for x in thickness_inner if x!=val]
    bins = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 5)) 
    weights_outer = np.ones_like(thickness_outer)/float(len(thickness_outer))
    plt.hist(thickness_outer, bins=bins, alpha=0.5, weights=weights_outer, label="Media") 
    weights_inner = np.ones_like(thickness_inner)/float(len(thickness_inner))
    plt.hist(thickness_inner, bins=bins, alpha=0.5, weights=weights_inner, label="Intima")
    plt.xticks(np.arange(0, 1.01, step=0.2), fontsize=15)
#     plt.yticks(np.arange(0, 0.13, step=0.02), fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel("Probability", fontsize=20)
    plt.legend(loc='upper right', ncol=1, fontsize=20)
    if path_to_svae:
        plt.savefig(path_to_svae)
    plt.show()
    

def clip_normalize(thick_media, thick_intima, thick_wall, plot_hist=False):
    assert len(thick_media) == len(thick_intima) == len(thick_wall) == 360
    
    # visualize hist before processing
#     if plot_hist:
#         plot_hist_w_two_list(thick_media, thick_intima, "Thickness", [-1], None)
    # Clip by 0.05 * median value of  thick_wall
    # remove all -1s, -1 means not intersection/thickness was found at this sample 
    clip_th = 0.05*np.median([x for x in thick_wall if x!= -1])
    thick_media_norm = [-1]*len(thick_media)
    thick_intima_norm = [-1]*len(thick_intima)
    for i in range(len(thick_wall)):
        if thick_media[i] <= clip_th or thick_intima[i] <= clip_th: 
            # if either media or intima thickness is below threshold, discard both
            continue
        else:
            thick_media_norm[i] = thick_media[i] / thick_wall[i]
            thick_intima_norm[i] = thick_intima[i] / thick_wall[i]
    thick_media_norm = [x for x in thick_media_norm if x!=-1]
    thick_intima_norm = [x for x in thick_intima_norm if x!=-1]
    if plot_hist:
        plot_hist_w_two_list(thick_media_norm, thick_intima_norm, 
                             "Normalizied Thickness", [], plot_hist)
    return thick_media_norm, thick_intima_norm

def get_features(dict_features, l_thick, prefix): 
    arr_thick = np.array(sorted(l_thick, reverse=True))
    # average of the top 5% value
    dict_features[prefix+" Average"] = np.mean(arr_thick)
    dict_features[prefix+" Skewness"] = scipy.stats.skew(arr_thick)
    dict_features[prefix+" Power"] = scipy.sum(arr_thick**2)/arr_thick.size