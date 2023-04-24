import numpy as np
from scipy.signal import find_peaks
from post_process_utils import *


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


def get_peak_prominences(data, peak_indices):
    n = len(data)
    prominences = []
    left_bases = []
    right_bases = []

    for peak_index in peak_indices:
        left_index = peak_index
        
        while left_index > 0 and ~np.isnan(data[left_index - 1]) and data[left_index - 1] < data[left_index]:
            left_index -= 1

        right_index = peak_index
        while right_index < n - 1 and ~np.isnan(data[right_index + 1]) and data[right_index + 1] < data[right_index]:
            right_index += 1

        left_base = data[left_index]
        right_base = data[right_index]
        if peak_index != left_index and peak_index != right_index:
            prominence = max(data[peak_index] - left_base, data[peak_index] - right_base)
        elif peak_index == left_index:
            prominence = data[peak_index] - right_base
        else:
            prominence = data[peak_index] - left_base
        
        prominences.append(prominence)
        left_bases.append(left_index)
        right_bases.append(right_index)

    return np.array(prominences), np.array(left_bases), np.array(right_bases)

def circular_peaks(data, distance=1, width_threshold=6):
    n = len(data)
    data_extended = np.concatenate([data, data, data])
    min_value = np.nanmin(data)
    data_extended_no_nan = np.where(np.isnan(data_extended), min_value - 1, data_extended)
    
    peak_indices_extended, _ = find_peaks(data_extended_no_nan, distance=distance)
    prominences_extended, left_bases_extended, right_bases_extended = get_peak_prominences(data_extended, peak_indices_extended)
    widths_extended = right_bases_extended - left_bases_extended
    
    peak_indices, prominences, widths, left_bases, right_bases = [], [], [], [], []
    for index, prominence, width, left_base, right_base in zip(peak_indices_extended, prominences_extended, widths_extended, left_bases_extended, right_bases_extended):
        if n <= index < 2 * n:
            if width_threshold is None or width >= width_threshold:
                peak_indices.append(index % n)
                prominences.append(prominence)
                widths.append(width)
                left_bases.append(left_base)
                right_bases.append(right_base)
    
    if len(peak_indices) == 0:
        return None, None

    peak_heights = data[np.array(peak_indices)]
    widths = np.array(right_bases) - np.array(left_bases) + 1

    peak_properties = {
        'peak_heights': peak_heights,
        'prominences': np.array(prominences),
        'widths': widths,
    }
    
    return np.array(peak_indices), peak_properties


def get_features(l_thick, prefix): 
    dict_features = {}
    peak_indices, peak_properties = circular_peaks(np.array([x if x >=0 else np.nan for x in l_thick]), width_threshold=15)
    if peak_indices is not None:
        dict_features[prefix+" Peak Height"] = np.max(peak_properties["peak_heights"])
        dict_features[prefix+" Peak Prominence"] = np.max(peak_properties["prominences"])
    else:
        dict_features[prefix+" Peak Height"] = 0
        dict_features[prefix+" Peak Prominence"] = 0
    
    l_thick_valid = [x for x in l_thick if x>=0]

    arr_thick = np.array(l_thick_valid)
    dict_features[prefix+" Average"] = np.mean(arr_thick)
    dict_features[prefix+" Median"] = np.median(arr_thick)
    dict_features[prefix+" Variance"] = np.var(arr_thick)
    dict_features[prefix+" Energy"] = loop_energy(l_thick_valid)
    return dict_features

def extract_features(thick_media, thick_intima, thick_wall):
        
    intima_media_ratio = [y/x if (x > 0 and y > 0) else 0 for x, y in zip(thick_media, thick_intima)]
    
    features_intima = get_features(thick_intima, "Intima")
    features_media = get_features(thick_media, "Media")
    features_ratio = get_features(intima_media_ratio, "Ratio")
    
    return features_intima, features_media, features_ratio
    
