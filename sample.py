def measure_thickness(wsi_id, artery_id, outer, middle, inner, vis, angle_width=15, exclude=[]):
    # close th contour for shapgeo
    outer = close_cnt(outer)
    middle = close_cnt(middle)
    inner = close_cnt(inner)
    
    # get centroid based on inner
    M = cv2.moments(inner)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
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
        curr_vis = vis_temp.copy()
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
        if insec_w_others: 
            continue
        
        dist_outer = euclidean(insec_mid, insec_outer) 
        dist_inner = euclidean(insec_mid, insec_inner)
                
        thickness_outer[i] = dist_outer
        thickness_inner[i] = dist_inner
        thickness_wall[i] = dist_outer + dist_inner
            
    return thickness_outer, thickness_inner, thickness_wall