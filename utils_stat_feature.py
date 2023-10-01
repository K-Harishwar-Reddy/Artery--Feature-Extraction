import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew

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
    l_thick_valid = [x for x in l_thick if x>=0]

    arr_thick = np.array(l_thick_valid)
    if prefix == "Ratio":
        arr_thick = arr_thick[arr_thick >= 0.05]
        arr_thick = arr_thick[arr_thick <= 20]
    dict_features[prefix+" Average"] = np.mean(arr_thick)
    dict_features[prefix+" Median"] = np.median(arr_thick)
    dict_features[prefix+" Variance"] = np.var(arr_thick)

    peak_indices, peak_properties = circular_peaks(np.array([x if x >=0 else np.nan for x in l_thick]), width_threshold=15)
    if peak_indices is not None:
        dict_features["Vis "+prefix+" Peak Indice"] = peak_indices[np.argmax(peak_properties["peak_heights"])]
        dict_features[prefix+" Peak Height"] = np.max(peak_properties["peak_heights"])
        dict_features[prefix+" Peak Prominence"] = np.max(peak_properties["prominences"])
    else:
        dict_features["Vis "+prefix+" Peak Indice"] = None
        dict_features[prefix+" Peak Height"] = 0
        dict_features[prefix+" Peak Prominence"] = 0
    return dict_features


def extract_features(thick_media, thick_intima, thick_ratio):
    
    features_intima = get_features(thick_intima, "Intima")
    features_media = get_features(thick_media, "Media")
    features_ratio = get_features(thick_ratio, "Ratio")
    
    return features_intima, features_media, features_ratio
    
