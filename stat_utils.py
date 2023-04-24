import numpy as np
import math

import scipy
from scipy import stats
from scipy.signal import find_peaks, peak_prominences, peak_widths

from matplotlib import pyplot as plt

def moving_window_remove(list_num, window_size=5, key = None, threshold=1):
    res = [None]*len(list_num)
    for i in range(len(list_num)):
        win = [list_num[(i - window_size // 2 + j) % len(list_num)] 
                   for j in range(window_size)]
        
        if win.count(key) >= threshold:
            res[i] = key
        else:
            res[i] = list_num[i]
    return np.array(res)

def moving_window_median(list_num, window_size=5):
    res = [None]*len(list_num)
    for i in range(len(list_num)):
        
        if list_num[i] >= 0:
            win = [list_num[(i - window_size // 2 + j) % len(list_num)] 
                       for j in range(window_size)]
            win = [x for x in win if x >= 0]
            res[i] = np.median(win)
        else:
            res[i] = list_num[i]
    return np.array(res)

def moving_window_average(list_num, window_size=5):
    res = [None]*len(list_num)
    for i in range(len(list_num)):
        if list_num[i] >= 0:
            win = [list_num[(i - window_size // 2 + j) % len(list_num)] 
                       for j in range(window_size)]
            win = [x for x in win if x >= 0]
            res[i] = np.mean(win)
        else:
            res[i] = list_num[i]
    return np.array(res)
    
def process_intersections(thick_media, thick_intima, thick_wall):
    thick_wall = moving_window_remove(thick_wall, window_size=61, key = -3, threshold=1)
    thick_media = moving_window_remove(thick_media, window_size=61, key = -3, threshold=1)
    thick_intima = moving_window_remove(thick_intima, window_size=61, key = -3, threshold=1)
    return thick_media, thick_intima, thick_wall

def process_open_lumens(thick_media, thick_intima, thick_wall):
    clip_th = 0.1*np.percentile([x for x in thick_wall if x>0], 75)
    idx = (thick_wall<clip_th) & (thick_wall >= 0)
    thick_wall[idx] = -2
    thick_media[idx] = -2
    thick_intima[idx] = -2
    thick_wall = moving_window_remove(thick_wall, window_size=61, key = -2, threshold=3)
    thick_media = moving_window_remove(thick_media, window_size=61, key = -2, threshold=3)
    thick_intima = moving_window_remove(thick_intima, window_size=61, key = -2, threshold=3)
    return thick_media, thick_intima, thick_wall

def process_moving_mediam(thick_media, thick_intima, thick_wall):
    thick_media = moving_window_median(thick_media, window_size=21)
    thick_intima = moving_window_median(thick_intima, window_size=21)
    thick_wall = moving_window_median(thick_wall, window_size=21)
    return thick_media, thick_intima, thick_wall

def process_moving_average(thick_media, thick_intima, thick_wall):
    thick_media = moving_window_average(thick_media, window_size=21)
    thick_intima = moving_window_average(thick_intima, window_size=21)
    thick_wall = moving_window_average(thick_wall, window_size=21)
    return thick_media, thick_intima, thick_wall

def process_impute(thick_media, thick_intima, thick_wall):
    thick_media = impute_missing_values(thick_media)
    thick_intima = impute_missing_values(thick_intima)
    thick_wall = impute_missing_values(thick_wall)
    return thick_media, thick_intima, thick_wall

def normalize(thick_media, thick_intima, thick_wall):
    base = np.median([x for x in thick_wall if x>0])
    thick_media = [x/base if x >=0 else x for x in thick_media]
    thick_intima = [x/base if x >=0 else x for x in thick_intima]    
    thick_wall = [x/base if x >=0 else x for x in thick_wall]
    return thick_media, thick_intima, thick_wall

def calculate_intima_media_ratio(thick_media, thick_intima):
    res = [y/x if (x > 0 and y > 0) else 0 for x, y in zip(thick_media, thick_intima)]
    
    return res

def find_closest_non_missing(lst, idx, direction):
    if direction == "left":
        step = -1
    elif direction == "right":
        step = 1
    else:
        raise ValueError("Invalid direction specified")
    
    current_idx = idx + step
    while 0 <= current_idx < len(lst):
        if lst[current_idx] >= 0:
            return current_idx, lst[current_idx]
        current_idx += step
    return None, None

def impute_missing_values(lst):
    for i, val in enumerate(lst):
        if val == -1:
            left_idx, left_val = find_closest_non_missing(lst, i, "left")
            right_idx, right_val = find_closest_non_missing(lst, i, "right")
            if left_val is not None and right_val is not None:
                weight_left = 1 / abs(left_idx - i)
                weight_right = 1 / abs(right_idx - i)
                lst[i] = (left_val * weight_left + right_val * weight_right) / (weight_left + weight_right)
            elif left_val is not None:
                lst[i] = left_val
            elif right_val is not None:
                lst[i] = right_val
    return lst

def plot_peaks(l_intima, l_media, l_ratio):
    colors = [(0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    plt.figure(figsize=(8, 4))
    for i, (x, y) in enumerate(zip([l_intima, l_media, l_ratio], ["Intima", "Media", "Intima-Media Ratio"])):
        peaks, properties = scipy.signal.find_peaks(
            x,
            prominence=0.1,
            height=0.5,
            width=None,
            distance=None
        )

        peak_heights = properties.get('peak_heights')

        if peak_heights is None:
            # Calculate peak heights if not available in properties
            peak_heights = np.array([x[p] for p in peaks])

        if len(peaks) == 0:
            continue

        highest_peak_index = np.argmax(peak_heights)
        peak_position = peaks[highest_peak_index]
        peak_height = peak_heights[highest_peak_index]
        plt.plot(x, color=colors[i], label=y)
        plt.xticks(fontsize=12)
        plt.plot(peak_position, np.array(x)[peak_position], color = colors[i], marker="x", markersize=12)
    plt.legend(fontsize=12)
    plt.show()    

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
#     ax2.set_ylabel('sin', color=color)
    ax2.plot(t, y_filter, color=color, label="Filtered")
    
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
#     plt.legend(fontsize="15")
#     ax1.legend(loc=0)
#     lns = lns1+lns2+lns3
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=2, fontsize=12)

    plt.show()


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
    

# def filter_signal_helper(x):
#     half_win = 15
#     res = [-1]*len(x)
#     for i in range(len(x)):
#         if i < half_win:
#             window = x[i-half_win+len(x):] + x[:i+half_win]
#         elif i > len(x)-half_win:
#             window = x[i-half_win:] + x[:i+half_win-len(x)]
#         else:
#             window = x[i-half_win:i+half_win]
#         if window.count(-1) >= half_win:
#             res[i] = -1
#         else:
#             window = [x for x in window if x != -1 ]
#             res[i] = np.mean(window)
        
#     return res
    
# def filter_signal(thick_media, thick_intima, thick_wall):
    
    
#     thick_media = filter_signal_helper(thick_media)
#     thick_intima = filter_signal_helper(thick_intima)
#     thick_wall = filter_signal_helper(thick_wall)
    
# #     idx = thick_wall.index(min(thick_wall))
# #     thick_media = thick_media[idx:] + thick_media[: idx]
# #     thick_intima = thick_intima[idx:] + thick_intima[: idx]
# #     thick_wall = thick_wall[idx:] + thick_wall[: idx]
    
#     thick_media = [x if x >= 0 else None for x in thick_media]
#     thick_intima = [x if x >= 0 else None for x in thick_intima]
#     thick_wall = [x if x >= 0 else None for x in thick_wall]
    
#     return thick_media, thick_intima, thick_wall
    
# def clip_normalize(thick_media, thick_intima, thick_wall, plot_hist=False):
#     assert len(thick_media) == len(thick_intima) == len(thick_wall) == 360
    
#     # visualize hist before processing
# #     if plot_hist:
# #         plot_hist_w_two_list(thick_media, thick_intima, "Thickness", [-1], None)
#     # Clip by 0.05 * median value of  thick_wall
#     # remove all -1s, -1 means not intersection/thickness was found at this sample 
#     clip_th = 0.05*np.median([x for x in thick_wall if x!= -1])
#     thick_media_norm = [-1]*len(thick_media)
#     thick_intima_norm = [-1]*len(thick_intima)
#     for i in range(len(thick_wall)):
#         if thick_media[i] < clip_th or thick_intima[i] < clip_th: 
#             # if either media or intima thickness is below threshold, discard both
#             continue
#         else:
#             thick_media_norm[i] = thick_media[i] / thick_wall[i]
#             thick_intima_norm[i] = thick_intima[i] / thick_wall[i]
#     thick_media_norm = [x for x in thick_media_norm if x!=-1]
#     thick_intima_norm = [x for x in thick_intima_norm if x!=-1]
#     if plot_hist:
#         plot_hist_w_two_list(thick_media_norm, thick_intima_norm, 
#                              "Normalizied Thickness", [], plot_hist)
#     return thick_media_norm, thick_intima_norm
def loop_energy(loop):
    energy = 0
    for i in range(len(loop) - 1):
        energy += (loop[i+1] - loop[i])**2
    energy += (loop[0] - loop[-1])**2  # Include the difference between the first and last elements
    return energy


def loop_entropy(loop, num_bins=10):
    min_value = min(loop)
    max_value = max(loop)
    
    # Bin the continuous values into discrete intervals
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    binned_values = np.digitize(loop, bin_edges)

    # Count the frequency of each bin
    element_counts = {}
    for element in binned_values:
        element_counts[element] = element_counts.get(element, 0) + 1

    # Calculate the entropy
    entropy = 0
    for count in element_counts.values():
        probability = count / len(loop)
        entropy -= probability * math.log2(probability)
    return entropy


def get_features(l_thick, prefix): 
    dict_features = {}
    peak_indices, peak_properties = circular_peaks(np.array([x if x >=0 else np.nan for x in l_thick]), width_threshold=15)
    if peak_indices is not None:
#         dict_features[prefix+" Peak Number"] = len(peak_indices)
        dict_features[prefix+" Peak Height"] = np.max(peak_properties["peak_heights"])
        dict_features[prefix+" Peak Prominence"] = np.max(peak_properties["prominences"])
    else:
#         dict_features[prefix+" Peak Number"] = 0
        dict_features[prefix+" Peak Height"] = 0
        dict_features[prefix+" Peak Prominence"] = 0
    
#     peak_height, peak_width = get_peak_features(l_thick)
    l_thick_valid = [x for x in l_thick if x>=0]

    arr_thick = np.array(l_thick_valid)
#     arr_thick = np.array(sorted(l_thick, reverse=True))
    # average of the top 5% value
    dict_features[prefix+" Average"] = np.mean(arr_thick)
    dict_features[prefix+" Median"] = np.median(arr_thick)
    dict_features[prefix+" Variance"] = np.var(arr_thick)
    dict_features[prefix+" Energy"] = loop_energy(l_thick_valid)
#     dict_features[prefix+" Entropy"] = loop_entropy(l_thick_valid)
#     dict_features[prefix+" Peak Height"] = peak_height
    return dict_features
