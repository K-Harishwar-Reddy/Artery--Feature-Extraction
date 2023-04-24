#!/usr/bin/python
# -*- coding: utf-8 -*-
# import necessary packages
import numpy as np
import cv2

ANN2RGB = {'Media': [[255, 0, 0]], 
           'Intima': [[49, 136, 235]],
           'Lumen': [[153, 102, 0]]}

# ANN2RGB = {'Media': [[255, 0, 0]], 
#            'Intima': [[102, 128, 230], [77, 102, 204], [49, 136, 235], [128, 153, 255], [128, 102, 204], 
#                      [49, 136, 235]],
#            'Lumen': [[255, 179, 102], [255, 204, 102], [204, 153, 51]]}

def count_lists(l):
    if not isinstance(l, list):
        return 0
    count = 0
    while isinstance(l, list):
        l = l[0]
        count += 1
    return count    

def clean_coord_heper(coords_raw, coords):
    if count_lists(coords_raw) == 2:
        coords += [coords_raw]
    else:
        for coords_raw_i in coords_raw: 
            clean_coord_heper(coords_raw_i, coords)

def clean_coord(coords_raw):
    coords_all = []
    clean_coord_heper(coords_raw, coords_all)
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
        coords_raw = ann_i['geometry']['coordinates']
        coords = clean_coord(coords_raw)
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

def cnt_polygon_test_2(cnt1, cnt2):
    # check if cnt1 inside/cross cnt2    
    for point in cnt1:        
        if cv2.pointPolygonTest(cnt2, (int(point[0]), int(point[1])), False) < 0: return True
    return False

def get_cnts_inside(ann, cnt_outer, target):
    cnts_inner_list = []
    for i, ann_i in enumerate(ann):
        ann_type = get_ann_type(ann_i)
        if ann_type == target:
            # check if inside or intersec
            cnt_inner_raw = ann_i["geometry"]["coordinates"]
            cnt_inner = clean_coord(cnt_inner_raw)
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