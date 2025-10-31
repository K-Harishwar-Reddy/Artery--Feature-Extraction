import numpy as np, cv2, os
import shapely.geometry as shapgeo
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
WSI_ARTERY_ID_FIGURE_MEASUREMENT = set()
WSI_ARTERY_ID_FIGURE_MISSING_VAL = set()
WSI_ARTERY_ID_FIGURE_LOCAL_FEATURES = set()
PEAK_IDX_COLOR = {}  # e.g., {angle:int -> (B,G,R)}

def count_lists(l):
    if not isinstance(l, list):
        return 0
    count = 0
    while isinstance(l, list):
        l = l[0]
        count += 1
    return count    

def clean_coord_helper(coords_raw, coords):
    if count_lists(coords_raw) == 2:
        coords += [coords_raw]
    else:
        for coords_raw_i in coords_raw: 
            clean_coord_helper(coords_raw_i, coords)

def clean_coord(coords_raw):
    coords_all = []
    clean_coord_helper(coords_raw, coords_all)
    max_length = len(coords_all[0])
    coord_res = coords_all[0]
    for i in range(1, len(coords_all)):
        if len(coords_all[i]) > max_length:
            max_length = len(coords_all[i])
            coord_res = coords_all[i]
    coord_res = np.array(coord_res, dtype=int)
    return coord_res
            
def get_border_of_cnt(cnt, border=0):
    # cnt: coordinates, list of list, [[0, 0], [0, 1], ...]
    cnt = np.array(cnt, dtype=int).squeeze()
    xmin, xmax = np.min(cnt[:, 0]), np.max(cnt[:, 0])
    ymin, ymax = np.min(cnt[:, 1]), np.max(cnt[:, 1])
    return (xmin-border, xmax+border, ymin-border, ymax+border)

def get_border_of_ann(ann, border=50):
    # ann: need to be cleaned to get coordinates
    (xmin, xmax) = (float('inf'), 0)
    (ymin, ymax) = (float('inf'), 0)
    for (i, ann_i) in enumerate(ann):
#         coords_raw = ann_i['geometry']['coordinates']
        coords = ann_i['geometry']['coordinates']
        curr_xmin, curr_xmax, curr_ymin, curr_ymax = get_border_of_cnt(coords)
        xmin = min(xmin, curr_xmin)
        xmax = max(xmax, curr_xmax)
        ymin = min(ymin, curr_ymin)
        ymax = max(ymax, curr_ymax)

    return (xmin-border, xmax+border, ymin-border, ymax+border)

def get_ann_type_by_color(color_rgb):
    for k in ANN2RGB:
        if color_rgb in ANN2RGB[k]:
            return k

def cnt_polygon_test(cnt1, cnt2):
    # check if cnt1 inside/cross cnt2    
    for point in cnt1:        
        if cv2.pointPolygonTest(cnt2, (int(point[0]), int(point[1])), False) >= 0: return True
    return False

def get_cnts_inside(ann, cnt_outer, target):
    cnts_inner_list = []
    for i, ann_i in enumerate(ann):
        ann_type = get_ann_type(ann_i)
        if ann_type == target:
            # check if inside or intersec
            cnt_inner = ann_i["geometry"]["coordinates"]
            if cnt_polygon_test(cnt_inner, cnt_outer):
                cnts_inner_list.append(cnt_inner)
    return cnts_inner_list
        
def adjust_artery_coords_by_boundry(cnt_outer, cnts_mid, cnts_inner, boundries):
    (xmin, _, ymin, _) = boundries
    cnt_outer[:, 0] = cnt_outer[:, 0] - xmin
    cnt_outer[:, 1] = cnt_outer[:, 1] - ymin
    
    for i in range(len(cnts_mid)):
        cnts_mid[i][:, 0] = cnts_mid[i][:, 0] - xmin
        cnts_mid[i][:, 1] = cnts_mid[i][:, 1] - ymin
        
    for i in range(len(cnts_inner)):
        cnts_inner[i][:, 0] = cnts_inner[i][:, 0] - xmin
        cnts_inner[i][:, 1] = cnts_inner[i][:, 1] - ymin   
    
    return cnt_outer, cnts_mid, cnts_inner
    
def get_ann_type(ann_i):
    if "classification" in ann_i["properties"] and "name" in ann_i["properties"]["classification"]:
        return ann_i["properties"]["classification"]["name"]
    else:
        print("STH WRONG")
        return None # noise

def get_cnt_idx_w_largest_area(cnts):
    max_area = 0
    max_area_idx = 0
    for i in range(len(cnts)):
        curr_area = cv2.contourArea(cnts[i])
        if curr_area > max_area:
            max_area = curr_area
            max_area_idx = i
    return max_area_idx

def get_centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

def euclidean(x, y):
    x = np.array(x).reshape(-1, 2).astype("float")
    y = np.array(y).reshape(-1, 2).astype("float")
    sum_sq = np.sum(np.square(x - y), axis=1)
    return np.sqrt(sum_sq) if len(sum_sq) > 1 else np.sqrt(sum_sq[0])

def find_insec_line_cnt(point1, point2, poly_contour):
    # Get the intersection between line segment [point1, point2] and poly_contour
    poly_line = shapgeo.LineString([point1, point2])
    intersections = poly_contour.intersection(poly_line)    
    return intersections

def find_insec_ray_cnt(start_pt, direction, poly_contour):   
    # Find maximum distance from start_pt to any point on poly_contour
    max_distance = 100000
    # Get the intersection between ray (start, direction) and poly_contour
    if isinstance(direction, (tuple, list)):
        # Direction can be a vector
        vx, vy = direction
    else:
        # Direction can be an angle (in degrees)
        angle = direction / 180 * np.pi
        vx, vy = np.cos(angle), np.sin(angle)
     
    
    x = int(vx * max_distance + start_pt[0])
    y = int(vy * max_distance + start_pt[1])
    return find_insec_line_cnt(start_pt, (x, y), poly_contour)

def get_furthest_closest(start_pt, points, metric):
    
    distances = [np.linalg.norm(point - start_pt) for point in points]
    if metric == 'farthest':
        index = np.argmax(distances)
    elif metric == 'closest':
        index = np.argmin(distances)
    return points[index]

def get_points_arr_from_shapgeo_insecs(insecs, start_pt):
    if isinstance(insecs, shapgeo.Point):
        return np.array(insecs.coords).reshape(-1, 2)
    elif isinstance(insecs, shapgeo.LineString):
        # get the point cloest to start point
        points = np.array(insecs.coords).reshape(-1, 2)
        if points.shape[0] <= 1:
            return points
        else:
            res = get_furthest_closest(start_pt, points, "closest")    
            return res.reshape(-1, 2)   
    elif isinstance(insecs, (shapgeo.MultiPoint, shapgeo.MultiLineString, shapgeo.GeometryCollection)):
        res = []
        for x in insecs.geoms:
            res.append(get_points_arr_from_shapgeo_insecs(x, start_pt))
        return np.vstack(res)
    else:
        print(type(insecs))

def find_insec_ray_cnt_w_filter(start_pt, direction, poly_contour, metric_if_multiple):
    insecs = find_insec_ray_cnt(start_pt, direction, poly_contour)
    insecs_arr = get_points_arr_from_shapgeo_insecs(insecs, start_pt)
    # If multiple intersection points are found, return the furthest/closest one
    if insecs_arr.shape[0] > 1:
        return get_furthest_closest(start_pt, insecs_arr, metric_if_multiple)
    elif insecs_arr.shape[0] == 1:
        return insecs_arr[0]
    else:
        return None

def get_insec_from_centre_to_poly(start_pt, angle, poly):
    # Get intersection of ray from (cx, cy) with angle and poly
    insec_point = find_insec_ray_cnt_w_filter(start_pt, angle, poly, "farthest")

    # If no intersection found, it means the centroid is outside of the poly,
    # so change to the other direction
    if insec_point is None:
        insec_point = find_insec_ray_cnt_w_filter(start_pt, (angle + 180) % 360, poly, "closest")
    return insec_point

def get_perp(point_start, point_end):
    # return the vector perpendicur to vector of (point_start, point_end)
    v_x = point_end[0] - point_start[0]
    v_y = point_end[1] - point_start[1]
    mag = np.sqrt(v_x * v_x + v_y * v_y)
    v_x = v_x / mag
    v_y = v_y / mag
    (v_x, v_y) = (v_y, -v_x)
    return (v_x, v_y)
    
def measure_thickness_per_angle(start_pt, angle, poly_outer, poly_middle, poly_inner, wsi_id, artery_id,
                                angle_width=15, exclude=[], vis=None, dir_parent="."):
    
    wsi_artery_id = wsi_id+"_"+artery_id
    if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MEASUREMENT and angle%70==0:
        vis = vis.copy()
        path_to_save = os.path.join(dir_parent, wsi_id, artery_id, wsi_id+"_"+artery_id+"_"+str(angle).zfill(3)+".png")
    
    # get insec of ray from (cx, cy) w angle and poly_middle
    insec_mid = get_insec_from_centre_to_poly(start_pt, angle, poly_middle)
    insec_mid_bef = get_insec_from_centre_to_poly(start_pt, angle - angle_width, poly_middle)
    insec_mid_aft = get_insec_from_centre_to_poly(start_pt, angle + angle_width, poly_middle)
    
    # get vector perpendicular to the tangent line
    (vx_outer, vy_outer) = get_perp(insec_mid_bef, insec_mid_aft)
    
    # insec with outer
    insec_outer = find_insec_ray_cnt_w_filter(insec_mid, (vx_outer, vy_outer), poly_outer, "closest")    
    if insec_outer is None: # Case of missing values
        if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MEASUREMENT and angle%70==0:
            vis_angle_discarded(vis, start_pt, insec_mid, path_to_save)
        if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MISSING_VAL:
            vis_angle_missing(vis, start_pt, insec_mid)
        return -1, -1
    
    (vx_inner, vy_inner) = get_perp(insec_mid_aft, insec_mid_bef)
    # insec with inner, more than one point should be found, 
    insec_inner = find_insec_ray_cnt(insec_mid, (vx_inner, vy_inner), poly_inner)
    insec_inner = get_points_arr_from_shapgeo_insecs(insec_inner, insec_mid)
    if insec_inner.shape[0] <= 1:
        if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MEASUREMENT and angle%70==0:
            vis_angle_discarded(vis, start_pt, insec_mid, path_to_save)
        if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MISSING_VAL:
            vis_angle_missing(vis, start_pt, insec_mid)
        return -1, -1
    else:
        insec_inner = get_furthest_closest(insec_mid, insec_inner, "closest")
    
    line_seg_outer = shapgeo.LineString([(insec_mid[0], insec_mid[1]), (insec_outer[0], insec_outer[1])])
    line_seg_inner = shapgeo.LineString([(insec_mid[0], insec_mid[1]), (insec_inner[0], insec_inner[1])])

    insec_w_others = False
    # get insec of ray from (cx, cy) w angle and poly_middle
    insec_outer_ray = get_insec_from_centre_to_poly(start_pt, angle, poly_outer)
    line_seg_ray = shapgeo.LineString([start_pt, (insec_outer_ray[0], insec_outer_ray[1])])
    for cnt in exclude:
        insec_seg_ray = shapgeo.LineString(cnt).intersects(line_seg_ray)
        insec_seg_outer = shapgeo.LineString(cnt).intersects(line_seg_outer)
        insec_seg_inner = shapgeo.LineString(cnt).intersects(line_seg_inner)
        if insec_seg_ray or insec_seg_outer or insec_seg_inner:
            insec_w_others = True
    if insec_w_others:
        if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MEASUREMENT and angle%70==0:
            vis_angle_discarded(vis, start_pt, insec_mid, path_to_save)
        return -3, -3
        
    dist_outer = euclidean(insec_mid, insec_outer) 
    dist_inner = euclidean(insec_mid, insec_inner)
        
    if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MEASUREMENT and angle%70==0:
        vis_angle_measurement(vis, start_pt, 
                          insec_inner, insec_mid, insec_outer, 
                          insec_mid_bef, insec_mid_aft, 
                          dist_inner, dist_outer, path_to_save)
        
    if wsi_artery_id in WSI_ARTERY_ID_FIGURE_LOCAL_FEATURES and angle in PEAK_IDX_COLOR:
        draw_line(vis, start_pt, insec_mid, PEAK_IDX_COLOR[angle])
        
    return dist_outer, dist_inner

def close_cnt(cnt):
    # close the cnt
    if np.array_equal(cnt[0], cnt[-1]):
        return cnt
    else:
        return np.vstack([cnt, cnt[0]])

def measure_thickness(cnt_outer, cnt_middle, cnt_inner, wsi_id, artery_id, angle_width=15, exclude=[], vis=None, dir_parent=None):
    # Assert contours are closed
    cnt_outer = close_cnt(cnt_outer)
    cnt_middle = close_cnt(cnt_middle)
    cnt_inner = close_cnt(cnt_inner)
    
    # Get the centroid
    cx, cy = get_centroid(cnt_inner)
    if abs(cv2.pointPolygonTest(cnt_middle, (cx,cy), True)) < 1:
        while abs(cv2.pointPolygonTest(cnt_middle, (cx,cy), True)) < 1:
            cx, cy = cx-1, cy

    # Set up angles (in degrees)
    angles = np.arange(0, 360, 1)

    # Prepare calculating the intersections using Shapely
    poly_outer = shapgeo.LineString(cnt_outer)
    poly_middle = shapgeo.LineString(cnt_middle)
    poly_inner = shapgeo.LineString(cnt_inner)

    thickness_outer = [None]*360
    thickness_inner = [None]*360

    for (i, angle) in enumerate(angles):
        dist_outer, dist_inner = measure_thickness_per_angle(
            (cx, cy), angle, poly_outer, poly_middle, poly_inner, wsi_id, artery_id, angle_width, exclude, vis, dir_parent)
        thickness_outer[i], thickness_inner[i] = dist_outer, dist_inner
    
    wsi_artery_id = wsi_id+"_"+artery_id
    # if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MISSING_VAL+WSI_ARTERY_ID_FIGURE_LOCAL_FEATURES:
    #     path_to_save = os.path.join(dir_parent, wsi_id, wsi_id+"_"+artery_id+'_ann.png')
    #     save_img_helper(vis, path_to_save)
    
    return thickness_outer, thickness_inner

def draw_line(image, start, end, color, thickness=2):
    """Utility function to draw a line on the image."""
    cv2.line(image, (int(start[0]), int(start[1])), 
                    (int(end[0]), int(end[1])), 
                    color, thickness)

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.8, color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA):
    """Utility function to draw text on the image."""
    cv2.putText(image, text, position, font, scale, color, thickness, line_type)

def vis_angle_discarded(vis, start_pt, insec_mid, path_to_save):
    draw_line(vis, start_pt, insec_mid, (255, 255, 255))    
    draw_text(vis, "discard", (10, 27))
    save_img_helper(vis, path_to_save)

def vis_angle_measurement(vis, start_pt, 
                          insec_inner, insec_mid, insec_outer, 
                          insec_mid_bef, insec_mid_aft, 
                          dist_inner, dist_outer, path_to_save):
    draw_line(vis, start_pt, insec_mid, (255, 255, 255))
    draw_line(vis, start_pt, insec_mid_bef, (128, 128, 128))
    draw_line(vis, start_pt, insec_mid_aft, (128, 128, 128))
    draw_line(vis, insec_inner, insec_mid, (255, 0, 255))
    draw_line(vis, insec_outer, insec_mid, (0, 255, 255))

    draw_text(vis, "intima: " + str(format(dist_inner, ".1f")), (10, 27), color=(255, 0, 255))
    draw_text(vis, "media: " + str(format(dist_outer, ".1f")), (10, 54), color=(0, 255, 255))
    save_img_helper(vis, path_to_save)
    
def vis_angle_missing(vis, start_pt, insec_mid):
    draw_line(vis, start_pt, insec_mid, (255, 255, 255))
        
def save_img_helper(img, path_to_save):
    Path(path_to_save).parents[0].mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path_to_save)
    
def plot_artery_ann(vis, cnt_outer, cnts_mid, cnts_inner, cnt_thick=2):
    cv2.drawContours(vis, [cnt_outer], -1, [255, 0, 0], cnt_thick)
    cv2.drawContours(vis, cnts_mid, -1, [0, 255, 0], cnt_thick)
    cv2.drawContours(vis, cnts_inner, -1, [0, 0, 255], cnt_thick)
    return vis
    
def save_img_for_seg(img, parent_dir, wsi_id, artery_id, category):
    dir_save = os.path.join(parent_dir, category)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    path_to_svae = os.path.join(dir_save, wsi_id+'_'+artery_id+'.png')
    Image.fromarray(img).save(path_to_svae)

def imshow_k_in_row(list_arr):
    k = len(list_arr)
    plt.figure(figsize=(5 * k, 5))  
    for i in range(k):
        plt.subplot(1, k, i + 1)
        plt.imshow(list_arr[i], cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def moving_window_remove(list_num, window_size=10, key = None, threshold=1):
    res = [None]*len(list_num)
    for i in range(len(list_num)):
        win = [list_num[(i - window_size // 2 + j) % len(list_num)] 
                   for j in range(window_size)]
        if win.count(key) >= threshold:
            res[i] = key
        else:
            res[i] = list_num[i]
    return np.array(res)

def moving_window_median(list_num, window_size=10):
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

def moving_window_average(list_num, window_size=10):
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
    
def process_intersections(thick_media, thick_intima, thick_wall, win_size):
    thick_wall = moving_window_remove(thick_wall, window_size=win_size, key = -3, threshold=1)
    thick_media = moving_window_remove(thick_media, window_size=win_size, key = -3, threshold=1)
    thick_intima = moving_window_remove(thick_intima, window_size=win_size, key = -3, threshold=1)
    return thick_media, thick_intima, thick_wall

def process_open_lumens(thick_media, thick_intima, thick_wall, win_size):
    clip_th = 0.1*np.percentile([x for x in thick_wall if x>0], 75)
    idx = (thick_wall<clip_th) & (thick_wall >= 0)
    thick_wall[idx] = -2
    thick_media[idx] = -2
    thick_intima[idx] = -2
    thick_wall = moving_window_remove(thick_wall, window_size=win_size, key = -2, threshold=5)
    thick_media = moving_window_remove(thick_media, window_size=win_size, key = -2, threshold=5)
    thick_intima = moving_window_remove(thick_intima, window_size=win_size, key = -2, threshold=5)
    return thick_media, thick_intima, thick_wall

def process_moving_mediam(thick_media, thick_intima, thick_wall, win_size):
    thick_media = moving_window_median(thick_media, window_size=win_size)
    thick_intima = moving_window_median(thick_intima, window_size=win_size)
    thick_wall = moving_window_median(thick_wall, window_size=win_size)
    return thick_media, thick_intima, thick_wall

def process_moving_average(thick_media, thick_intima, thick_wall, win_size):
    thick_media = moving_window_average(thick_media, window_size=win_size)
    thick_intima = moving_window_average(thick_intima, window_size=win_size)
    thick_wall = moving_window_average(thick_wall, window_size=win_size)
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

def post_process(thick_media, thick_intima, thick_wall, t_multi=11, t_open_lumen=11, t_mediam=11, t_average=11):
    thick_media, thick_intima, thick_wall = process_intersections(thick_media, thick_intima, thick_wall, t_multi)
    thick_media, thick_intima, thick_wall = process_open_lumens(thick_media, thick_intima, thick_wall, t_open_lumen)
    thick_media, thick_intima, thick_wall = process_moving_mediam(thick_media, thick_intima, thick_wall, t_mediam)
    thick_media, thick_intima, thick_wall = process_moving_average(thick_media, thick_intima, thick_wall, t_average)    
    thick_media, thick_intima, thick_wall = process_impute(thick_media, thick_intima, thick_wall)
    thick_media, thick_intima, thick_wall = normalize(thick_media, thick_intima, thick_wall)
    thick_ratio = [y/(x+y) if (x > 0 and y > 0) else 0 for x, y in zip(thick_media, thick_intima)]
    return thick_media, thick_intima, thick_ratio

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

# def get_features(l_thick, prefix): 
#     dict_features = {}
#     l_thick_valid = [x for x in l_thick if x>=0]

#     arr_thick = np.array(l_thick_valid)
#     if prefix == "Ratio":
#         arr_thick = arr_thick[arr_thick >= 0.05]
#         arr_thick = arr_thick[arr_thick <= 20]
#     dict_features[prefix+" Average"] = np.mean(arr_thick)
#     dict_features[prefix+" Median"] = np.median(arr_thick)
#     dict_features[prefix+" Variance"] = np.var(arr_thick)
#     dict_features[prefix + " Skewness"]  = float(skew(arr, bias=False, nan_policy='omit')) if arr.size >= 3 else np.nan
#     dict_features[prefix + " Kurtosis"]  = float(kurtosis(arr, fisher=True, bias=False, nan_policy='omit')) if arr.size >= 4 else np.nan

#     peak_indices, peak_properties = circular_peaks(np.array([x if x >=0 else np.nan for x in l_thick]), width_threshold=15)
#     if peak_indices is not None:
#         dict_features["Vis "+prefix+" Peak Indice"] = peak_indices[np.argmax(peak_properties["peak_heights"])]
#         dict_features[prefix+" Peak Height"] = np.max(peak_properties["peak_heights"])
#         dict_features[prefix+" Peak Prominence"] = np.max(peak_properties["prominences"])
#     else:
#         dict_features["Vis "+prefix+" Peak Indice"] = None
#         dict_features[prefix+" Peak Height"] = 0
#         dict_features[prefix+" Peak Prominence"] = 0
#     return dict_features
from scipy.stats import skew, kurtosis

def get_features(l_thick, prefix):
    dict_features = {}

    # keep valid values
    arr_thick = np.asarray([x for x in l_thick if x >= 0], dtype=float)
    if prefix == "Ratio":
        arr_thick = arr_thick[(arr_thick >= 0.05) & (arr_thick <= 20)]

    if arr_thick.size == 0:
        dict_features[prefix+" Average"]   = np.nan
        dict_features[prefix+" Median"]    = np.nan
        dict_features[prefix+" Variance"]  = np.nan
        dict_features[prefix+" Skewness"]  = np.nan
        dict_features[prefix+" Kurtosis"]  = np.nan
    else:
        dict_features[prefix+" Average"]   = float(np.mean(arr_thick))
        dict_features[prefix+" Median"]    = float(np.median(arr_thick))
        dict_features[prefix+" Variance"]  = float(np.var(arr_thick))
        dict_features[prefix+" Skewness"]  = float(skew(arr_thick, bias=False, nan_policy='omit')) if arr_thick.size >= 3 else np.nan
        dict_features[prefix+" Kurtosis"]  = float(kurtosis(arr_thick, fisher=True, bias=False, nan_policy='omit')) if arr_thick.size >= 4 else np.nan

    arr_nan = np.array([x if x >= 0 else np.nan for x in l_thick], dtype=float)
    peak_indices, peak_properties = circular_peaks(arr_nan, width_threshold=15)
    if peak_indices is not None:
        dict_features["Vis "+prefix+" Peak Indice"] = int(peak_indices[np.argmax(peak_properties["peak_heights"])])
        dict_features[prefix+" Peak Height"]        = float(np.max(peak_properties["peak_heights"]))
        dict_features[prefix+" Peak Prominence"]    = float(np.max(peak_properties["prominences"]))
    else:
        dict_features["Vis "+prefix+" Peak Indice"] = None
        dict_features[prefix+" Peak Height"]        = 0.0
        dict_features[prefix+" Peak Prominence"]    = 0.0

    return dict_features

def extract_features(thick_media, thick_intima, thick_ratio):
    
    features_intima = get_features(thick_intima, "Intima")
    features_media = get_features(thick_media, "Media")
    features_ratio = get_features(thick_ratio, "Ratio")
    
    return features_intima, features_media, features_ratio
    

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
            ax1.axvline(x=x,  linestyle="--", color="gray", alpha=0.5, label="Missing")
        else:
            ax1.axvline(x=x,  linestyle="--", color="gray", alpha=0.5)

    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2 = ax1.twinx()
    color = 'darkorange'
    ax2.plot(t, y_filter, color=color, label="Processed")
    
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=2, fontsize=12)
    plt.show()
    
def plot_local_peaks(thick_media, thick_intima, thick_ratio, p_idx_intima, p_idx_media, p_idx_ratio):
    plt.figure(figsize=(10, 5)) 
    color_intima, color_media, color_ratio = "#800080", "#008080", "#808000"
    
    plt.plot([x if x>=0 else None for x in thick_intima], color=color_intima, label="Intima")
    plt.plot([x if x>=0 else None for x in thick_media], color=color_media, label="Media")
    plt.plot([x if x>=0 else None for x in thick_ratio], color=color_ratio, label="Intima-Media Ratio")
    
    plt.scatter(p_idx_intima, np.array(thick_intima)[p_idx_intima], color=color_intima, marker="x", s=100)
    plt.scatter(p_idx_media, np.array(thick_media)[p_idx_media], color=color_media, marker="x", s=100)
    plt.scatter(p_idx_ratio, np.array(thick_ratio)[p_idx_ratio], color=color_ratio, marker="x", s=100)
    
    plt.xlabel("Angle", fontsize=20)
    plt.ylabel("Measurements", fontsize=20)
    plt.xticks(np.arange(0, 361, step=120), fontsize=15)
    plt.legend(fontsize=20, framealpha=0.5, loc='upper right')
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
    plt.ylabel("Thickness", fontsize=20)
    plt.legend(loc='upper right', ncol=1, fontsize=20)
    if path_to_svae:
        plt.savefig(path_to_svae)
    plt.show()