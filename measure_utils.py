#!/usr/bin/python
# -*- coding: utf-8 -*-
# import necessary packages

import numpy as np
import cv2
import shapely
import shapely.geometry as shapgeo
from vis_utils import save_img_animation_helper
import sympy as sym
import sympy.printing as printing
import itertools

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

    
def measure_thickness(wsi_id, artery_id, outer, middle, inner, vis, angle_width=15, exclude=[]):
    vis_temp = vis.copy()
    interval = 72

    # close th contour for shapgeo
    outer = close_cnt(outer)
    middle = close_cnt(middle)
    inner = close_cnt(inner)
    
    # get centroid based on inner
    M = cv2.moments(inner)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
#     cv2.circle(vis, (cx, cy), radius=3, color=(255, 0, 0), thickness=-1)
    
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

    thickness_outer = [None]*360
    thickness_inner = [None]*360
    thickness_wall = [None]*360
    thick_plot = 2
    for (i, angle) in enumerate(angles):
        
        curr_vis = vis_temp.copy()
        # get insec of ray from (cx, cy) w angle and poly_middle
        insec_mid = find_insec((cx, cy), angle, r_max, poly_middle, "farthest")
        # if not insec found, it means the centroid is out of the middle, should change to the other direction
        if len(insec_mid) == 0:
            insec_mid = find_insec((cx, cy), (angle + 180) % 360, r_max, poly_middle, "closest")
            
        insec_outer_ray = find_insec((cx, cy), angle, r_max, poly_outer, "farthest")
        # if not insec found, it means the centroid is out of the middle, should change to the other direction
        if len(insec_outer_ray) == 0:
            insec_outer_ray = find_insec((cx, cy), (angle + 180) % 360, r_max, poly_outer, "closest")
        
        
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
        if len(insec_outer) == 0:
            thickness_outer[i] = -1
            thickness_inner[i] = -1
            thickness_wall[i] = -1
            continue
        
        (vx_inner, vy_inner) = get_perp(insec_mid_aft, insec_mid_bef)
        # insec with inner, more than one point should be found, 
        insec_inner = find_insec_helper(insec_mid, (vx_inner, vy_inner), r_max, poly_inner)
        insec_inner = insec_inner.reshape(-1, 2)
        if insec_inner.shape[0] <= 1:
            thickness_outer[i] = -1
            thickness_inner[i] = -1
            thickness_wall[i] = -1
            continue
        else:
            insec_inner = get_furthest_closest(insec_mid, insec_inner, "closest")            
        
        line_seg_outer = shapgeo.LineString([(insec_mid[0], insec_mid[1]), (insec_outer[0], insec_outer[1])])
        line_seg_inner = shapgeo.LineString([(insec_mid[0], insec_mid[1]), (insec_inner[0], insec_inner[1])])
        line_seg_ray = shapgeo.LineString([(cx, cy), (insec_outer_ray[0], insec_outer_ray[1])])
        
        # line segment is [insec_outer, insec_mid] should not intersect exclude
#         insec_inner = find_insec_helper(insec_mid, (vx_inner, vy_inner), r_max, poly_inner)
        
        insec_w_others = False
        for cnt in exclude:
            insec_seg_ray = shapgeo.LineString(cnt).intersects(line_seg_ray)
            insec_seg_outer = shapgeo.LineString(cnt).intersects(line_seg_outer)
            insec_seg_inner = shapgeo.LineString(cnt).intersects(line_seg_inner)
            if insec_seg_outer or insec_seg_inner:
                insec_w_others = True
        if insec_w_others:
            thickness_outer[i] = -3
            thickness_inner[i] = -3
            thickness_wall[i] = -3
            continue
        
        dist_outer = euclidean(insec_mid, insec_outer) 
        dist_inner = euclidean(insec_mid, insec_inner)
                
        thickness_outer[i] = dist_outer
        thickness_inner[i] = dist_inner
        thickness_wall[i] = dist_outer + dist_inner
        
        # for vis
        thick_vis = 2
        
#         if dist_inner > 0 and dist_outer > 0: # and i in [110]:
# #             if i == 43:
# #                 cv2.line(vis, (int(cx), int(cy)), (int(insec_mid[0]),
# #                          int(insec_mid[1])), (0, 128, 128), thick_vis)
# #             if i == 227:
# #                 cv2.line(vis, (int(cx), int(cy)), (int(insec_mid[0]),
# #                          int(insec_mid[1])), (128, 0, 128), thick_vis)
# #             if i == 89:
# #                 cv2.line(vis, (int(cx), int(cy)), (int(insec_mid[0]),
# #                          int(insec_mid[1])), (128, 128, 0), thick_vis)                
# #             cv2.line(vis, (int(cx), int(cy)), (int(insec_mid_bef[0]),
# #                      int(insec_mid_bef[1])), (128, 128, 128), thick_vis)
# #             cv2.line(vis, (int(cx), int(cy)), (int(insec_mid_aft[0]),
# #                      int(insec_mid_aft[1])), (128, 128, 128), thick_vis)
#             cv2.line(vis, (int(insec_inner[0]), int(insec_inner[1])), 
#                      (int(insec_mid[0]), int(insec_mid[1])), 
#                      (255, 0, 255), thick_vis)
#             cv2.line(vis, (int(insec_outer[0]), int(insec_outer[1])), 
#                      (int(insec_mid[0]), int(insec_mid[1])), 
#                      (0, 255, 255), thick_vis)
        
#                 # for vis
        

#         thick_plot = 2
#         clip_th = 3.085553431830905
        
#         if dist_inner > 0 and dist_outer > 0 and i%interval==0: #and dist_inner >= clip_th and dist_outer >= clip_th: #or i==120 or i==240):
#             cv2.line(curr_vis, (int(insec_mid_bef[0]), int(insec_mid_bef[1])), 
#                      (int(insec_mid_aft[0]), int(insec_mid_aft[1])), (255, 255, 255), thick_plot)
#             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid[0]),
#                      int(insec_mid[1])), (255, 255, 255), 2)    
#             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid_bef[0]),
#                      int(insec_mid_bef[1])), (128, 128, 128), 2)
#             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid_aft[0]),
#                      int(insec_mid_aft[1])), (128, 128, 128), 2)
#             cv2.line(curr_vis, (int(insec_inner[0]), int(insec_inner[1])), 
#                      (int(insec_mid[0]), int(insec_mid[1])), 
#                      (255, 0, 255), thick_plot)
#             cv2.line(curr_vis, (int(insec_outer[0]), int(insec_outer[1])), 
#                      (int(insec_mid[0]), int(insec_mid[1])),
#                      (0, 255, 255), thick_plot)
#             dist_inner, dist_outer = dist_inner/thickness_wall[i], dist_outer/thickness_wall[i]

#             cv2.putText(curr_vis, "intima: "+str(format(dist_inner, ".1f")), (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2, cv2.LINE_AA)
#             cv2.putText(curr_vis, "media: "+str(format(dist_outer, ".1f")), (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
        
#             cv2.putText(curr_vis, "intima: "+str(format(dist_inner, ".1f")), (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1, cv2.LINE_AA)
#             cv2.putText(curr_vis, "media: "+str(format(dist_outer, ".1f")), (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
#             save_img_animation_helper(curr_vis, wsi_id, artery_id, i) 
#         elif i%interval==0:
#             cv2.line(curr_vis, (int(cx), int(cy)), (int(insec_mid[0]), int(insec_mid[1])), (255, 255, 255), 2) 
#             cv2.putText(curr_vis, "discard", (8, 28), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
#             save_img_animation_helper(curr_vis, wsi_id, artery_id, i) 
           
    return thickness_outer, thickness_inner, thickness_wall


def get_centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy
    
def find_new_point(p, closest_p, contour):
    direction = np.array(closest_p) - np.array(p)
    unit_direction = direction / np.linalg.norm(direction)
    distance = euclidean(p, closest_p)
    for step_size in range(1, int(distance), 1):
        new_point = np.array(p) + step_size * unit_direction
        if cv2.pointPolygonTest(contour, tuple(map(int, new_point)), True) > 1:
            return tuple(map(int, new_point))
    return p

def adjust_cnt_inside(cnt_outer, cnt_middle, cnt_inner):
    centroid_middle = get_centroid(cnt_middle)
  
    for i, p in enumerate(cnt_middle):
        p = [int(p[0]), int(p[1])]
        if cv2.pointPolygonTest(cnt_outer, p, False) <= 1:
            cnt_middle[i] = find_new_point(p, centroid_middle, cnt_outer)
            
    centroid_inner = get_centroid(cnt_inner)
    for i, p in enumerate(cnt_inner):
        p = [int(p[0]), int(p[1])]
        if cv2.pointPolygonTest(cnt_middle, p, False) <= 1:
            cnt_inner[i] = find_new_point(p, centroid_inner, cnt_middle)

    return cnt_middle, cnt_inner