import numpy as np

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
    thick_wall = moving_window_remove(thick_wall, window_size=31, key = -3, threshold=1)
    thick_media = moving_window_remove(thick_media, window_size=31, key = -3, threshold=1)
    thick_intima = moving_window_remove(thick_intima, window_size=31, key = -3, threshold=1)
    return thick_media, thick_intima, thick_wall

def process_open_lumens(thick_media, thick_intima, thick_wall):
    clip_th = 0.1*np.percentile([x for x in thick_wall if x>0], 75)
    idx = (thick_wall<clip_th) & (thick_wall >= 0)
    thick_wall[idx] = -2
    thick_media[idx] = -2
    thick_intima[idx] = -2
    thick_wall = moving_window_remove(thick_wall, window_size=31, key = -2, threshold=5)
    thick_media = moving_window_remove(thick_media, window_size=31, key = -2, threshold=5)
    thick_intima = moving_window_remove(thick_intima, window_size=31, key = -2, threshold=5)
    return thick_media, thick_intima, thick_wall

def process_moving_mediam(thick_media, thick_intima, thick_wall):
    thick_media = moving_window_median(thick_media, window_size=11)
    thick_intima = moving_window_median(thick_intima, window_size=11)
    thick_wall = moving_window_median(thick_wall, window_size=11)
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

def post_process(thick_media, thick_intima, thick_wall):
    thick_media, thick_intima, thick_wall = process_intersections(thick_media, thick_intima, thick_wall)
    thick_media, thick_intima, thick_wall = process_open_lumens(thick_media, thick_intima, thick_wall)
    thick_media, thick_intima, thick_wall = process_moving_mediam(thick_media, thick_intima, thick_wall)
    thick_media, thick_intima, thick_wall = process_moving_average(thick_media, thick_intima, thick_wall)    
    thick_media, thick_intima, thick_wall =  process_impute(thick_media, thick_intima, thick_wall)
    thick_media, thick_intima, thick_wall = normalize(thick_media, thick_intima, thick_wall)
    return thick_media, thick_intima, thick_wall