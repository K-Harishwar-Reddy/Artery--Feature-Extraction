import os
import numpy as np
import shapely.geometry as shapgeo
import cv2
from PIL import Image
from utils_measure_vis import *

WSI_ARTERY_ID_FIGURE_MEASUREMENT = ["12_26609_022_015 L2 TRI_A04", 
                                    "11_26609_000_011_L02_TRI_A11", 
                                    "KL-0023191_L03_TRIS_272723_A07"]

WSI_ARTERY_ID_FIGURE_MISSING_VAL = ["11_26609_000_011_L02_TRI_A01"]

WSI_ARTERY_ID_FIGURE_LOCAL_FEATURES = ["11_26609_009_008 L10 TRI_A47"]

PEAK_IDX_COLOR = {34: (128, 0, 128), 226: (0, 128, 128), 84: (128, 128, 0)}


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
    if wsi_artery_id in WSI_ARTERY_ID_FIGURE_MISSING_VAL+WSI_ARTERY_ID_FIGURE_LOCAL_FEATURES:
        path_to_save = os.path.join(dir_parent, wsi_id, wsi_id+"_"+artery_id+'_ann.png')
        save_img_helper(vis, path_to_save)
    
    return thickness_outer, thickness_inner