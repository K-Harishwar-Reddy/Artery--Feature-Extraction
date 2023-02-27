#!/usr/bin/python
# -*- coding: utf-8 -*-
# import necessary packages

import numpy as np
import cv2
import shapely
import shapely.geometry as shapgeo

def euclidean(x, y):
    x = np.array(x).reshape(-1, 2).astype("float")
    y = np.array(y).reshape(-1, 2).astype("float")
    sum_sq = np.sum(np.square(x - y), axis=1)
    return np.sqrt(sum_sq) if len(sum_sq) > 1 else np.sqrt(sum_sq[0])


def find_insec_line_cnt(point1, pount2, poly_contour):
    # get the intersection between line seg [p1, p2] with cnt
    poly_line = shapgeo.LineString([point1, pount2])
    intersections = poly_contour.intersection(poly_line)
    return intersections

    
def find_insec_helper(start_pt, direction, r_max, poly_contour):
    # get the intersection between ray (start, direction) with cnt
    if type(direction) == tuple or type(direction) == list:
        # direction can be a vector
        vx, vy = direction
    else:
        # direction can be an integer of angle
        angle = direction / 180 * np.pi
        vx, vy = np.cos(angle), np.sin(angle)
    cx, cy = start_pt
    x = vx * r_max + cx
    y = vy * r_max + cy
    
    insecs = find_insec_line_cnt((cx, cy), (x, y), poly_contour)
    if isinstance(insecs, shapely.geometry.point.Point) or isinstance(insecs,
            shapely.geometry.linestring.LineString):
        # if return type is point for linestring, 
        insecs_arr = np.array(insecs.coords).reshape(-1, 2)
    else:
        # return can be other geometry types where > two points returned.
        insecs_list = []
        for x in insecs.geoms:
            insecs_list.append(np.array(x.coords).reshape(-1, 2))
        insecs_arr = np.vstack(insecs_list)
    if insecs_arr.shape[0] == 0:
        return np.array([])
    elif insecs_arr.shape[0] == 1: 
        return insecs_arr[0]
    else:
        return insecs_arr

def find_insec(start_pt, direction, r_max, poly_contour, metric_if_multiple = 'farthest'):
    insecs_arr = find_insec_helper(start_pt, direction, r_max, poly_contour)
    if len(insecs_arr.shape) > 1:
        # when multiple points found, get the furthest/closest one.
        return get_furthest_closest(start_pt, insecs_arr, metric_if_multiple)
    else:
        return insecs_arr

def close_cnt(cnt):
    # close the cnt
    cnt = cnt.squeeze()
    cnt = np.vstack([cnt, cnt[0]])
    return cnt

def get_furthest_closest(point, candidates, case_multiple='farthest'):
    dists = euclidean(point, candidates)
    if case_multiple == 'farthest':
        return candidates[np.argmax(dists)]
    elif case_multiple == 'closest':
        return candidates[np.argmin(dists)] 


def get_perp(point_start, point_end):
    # return the vector perpendicur to vector of (point_start, point_end)
    v_x = point_end[0] - point_start[0]
    v_y = point_end[1] - point_start[1]
    mag = np.sqrt(v_x * v_x + v_y * v_y)
    v_x = v_x / mag
    v_y = v_y / mag
    (v_x, v_y) = (v_y, -v_x)
    return (v_x, v_y)

    
def measure_thickness(wsi_id, artery_id, outer, middle, inner, vis, angle_width=10, exclude=[]):
    
    # close th contour for shapgeo
    outer = close_cnt(outer)
    middle = close_cnt(middle)
    inner = close_cnt(inner)
    
    # get centroid based on inner
    M = cv2.moments(inner)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(vis, (cx, cy), radius=3, color=(255, 0, 0), thickness=-1)
    
    # Set up angles (in degrees)
    angles = np.arange(0, 360, 1)
    
    # Prepare calculating the intersections using Shapely
    poly_outer = shapgeo.LineString(outer)
    poly_middle = shapgeo.LineString(middle)
    poly_inner = shapgeo.LineString(inner)
    
    
    # Calculate maximum needed radius for later line intersections
    # this is to pseudo ray, when calcualting intersection
    (h, w) = vis.shape[:2]
    r_max = np.max([euclidean([cx, cy], [0, 0]), euclidean([cx, cy], [h
                   - 1, 0]), euclidean([cx, cy], [0, w - 1]),
                   euclidean([cx, cy], [h - 1, w - 1])])

    thickness_outer = [-1]*360
    thickness_inner = [-1]*360
    thickness_wall = [-1]*360
    for (i, angle) in enumerate(angles):
        # get insec of ray from (cx, cy) w angle and poly_middle
        insec_mid = find_insec((cx, cy), angle, r_max, poly_middle, "farthest")
        # if not insec found, it means the centroid is out of the middle, should change to the other direction
        if len(insec_mid) == 0:
            insec_mid = find_insec((cx, cy), (angle + 180) % 360, r_max, poly_middle, "closest")
        
        # find insec_mid_bef and insec_mid_aft to get the tangent line at angle
        insec_mid_bef = find_insec((cx, cy), angle - angle_width, r_max, poly_middle, "farthest")
        if len(insec_mid_bef) == 0:
            insec_mid_bef = find_insec((cx, cy), (angle - angle_width + 180) % 360, r_max, poly_middle, "closest")
        
        insec_mid_aft = find_insec((cx, cy), angle + angle_width, r_max, poly_middle, "farthest")
        if len(insec_mid_aft) == 0:
            insec_mid_aft = find_insec((cx, cy), (angle + angle_width + 180) % 360, r_max, poly_middle, "closest")
        # get vector perpendicular to the tangent line
        (vx_outer, vy_outer) = get_perp(insec_mid_bef, insec_mid_aft)
        # insec with outer
        insec_outer = find_insec(insec_mid, (vx_outer, vy_outer), r_max, poly_outer, "farthest")
        
        (vx_inner, vy_inner) = get_perp(insec_mid_aft, insec_mid_bef)
        # insec with inner, more than one point should be found, 
        insec_inner = find_insec_helper(insec_mid, (vx_inner, vy_inner), r_max, poly_inner)
        if len(insec_inner.shape) == 1:
            # case 1: no insec
            # case 2: only one found, means middle inside inner, wrong ann
            continue
        else:
            insec_inner = get_furthest_closest(insec_mid, insec_inner, "closest")
        if len(insec_outer) == 0 or len(insec_inner) == 0:
            continue
        
        line_seg_outer = shapgeo.LineString([(insec_mid[0], insec_mid[1]), (insec_outer[0], insec_outer[1])])
        line_seg_inner = shapgeo.LineString([(insec_mid[0], insec_mid[1]), (insec_inner[0], insec_inner[1])])
        
        # line segment is [insec_outer, insec_mid] should not intersect exclude
        insec_w_others = False
        for cnt in exclude:
            insec_seg_outer = shapgeo.LineString(cnt).intersects(line_seg_outer)
            insec_seg_inner = shapgeo.LineString(cnt).intersects(line_seg_inner)
            if insec_seg_outer or insec_seg_inner:
                insec_w_others = True
        if insec_w_others: continue
        
        dist_outer = euclidean(insec_mid, insec_outer) 
        dist_inner = euclidean(insec_mid, insec_inner)
                
        thickness_outer[i] = dist_outer
        thickness_inner[i] = dist_inner
        thickness_wall[i] = dist_outer + dist_inner
        
        # for vis
        thick_vis = 2
        if dist_inner > 0 and dist_outer > 0 and (i == 0): #or i==120 or i==240):
            cv2.line(vis, (int(cx), int(cy)), (int(insec_mid[0]),
                     int(insec_mid[1])), (255, 255, 255), thick_vis)
#             cv2.line(vis, (int(cx), int(cy)), (int(insec_mid_bef[0]),
#                      int(insec_mid_bef[1])), (128, 128, 128), thick_vis)
#             cv2.line(vis, (int(cx), int(cy)), (int(insec_mid_aft[0]),
#                      int(insec_mid_aft[1])), (128, 128, 128), thick_vis)
            cv2.line(vis, (int(insec_inner[0]), int(insec_inner[1])), 
                     (int(insec_mid[0]), int(insec_mid[1])), 
                     (255, 0, 255), thick_vis)
            cv2.line(vis, (int(insec_outer[0]), int(insec_outer[1])), 
                     (int(insec_mid[0]), int(insec_mid[1])), 
                     (0, 255, 255), thick_vis)
        
            
#                 # for vis
#         curr_vis = vis.copy()

#         thick_plot = 2
#         clip_th = 3.965279047150565
#         if dist_inner > 0 and dist_outer > 0 and dist_inner >= clip_th and dist_outer >= clip_th: #or i==120 or i==240):

            
#             cv2.line(curr_vis, (int(insec_mid_bef[0]), int(insec_mid_bef[1])), 
#                      (int(insec_mid_aft[0]), int(insec_mid_aft[1])), (255, 255, 255), thick_plot)
#             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid[0]),
#                      int(insec_mid[1])), (255, 255, 255), 1)    
    
# #             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid_bef[0]),
# #                      int(insec_mid_bef[1])), (128, 128, 128), 1)
# #             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid_aft[0]),
# #                      int(insec_mid_aft[1])), (128, 128, 128), 1)
#             cv2.line(curr_vis, (int(insec_inner[0]), int(insec_inner[1])), 
#                      (int(insec_mid[0]), int(insec_mid[1])), 
#                      (255, 0, 255), thick_plot)
#             cv2.line(curr_vis, (int(insec_outer[0]), int(insec_outer[1])), 
#                      (int(insec_mid[0]), int(insec_mid[1])), 
#                      (0, 255, 255), thick_plot)
#             dist_inner, dist_outer = dist_inner/thickness_wall[i], dist_outer/thickness_wall[i]
#             cv2.putText(curr_vis, "intima: "+str(format(dist_inner, ".2f")), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
#             cv2.putText(curr_vis, "media: "+str(format(dist_outer, ".2f")), (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
#             save_img_animation_helper(curr_vis, wsi_id, artery_id, i//10) 
#         else:
#             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid[0]), int(insec_mid[1])), (255, 255, 255), 1) 
#             cv2.putText(curr_vis, "skip", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
#             save_img_animation_helper(curr_vis, wsi_id, artery_id, i//10) 
            
    return thickness_outer, thickness_inner, thickness_wall