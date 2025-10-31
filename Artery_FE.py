import os, time, cv2, json, geojson, openslide, scipy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from helper import (adjust_artery_coords_by_boundry, cnt_polygon_test, clean_coord, 
                        get_border_of_ann, get_border_of_cnt, get_cnts_inside, measure_thickness, 
                    plot_artery_ann, save_img_helper, post_process,extract_features,
                    plot_hist_w_two_list, plot_raw_filter, plot_local_peaks)
from collections import Counter
import argparse
import time

def _hist_labels(ann):
    def _name(f):
        p = f.get('properties', {})
        # Try common QuPath shapes:
        if isinstance(p.get('classification'), dict) and 'name' in p['classification']:
            return str(p['classification']['name'])
        # fallback
        return str(p.get('name', ''))
    return Counter(_name(f) for f in ann)

def _first_polys_info(ann, k=5):
    # peek first k features with a few coords and bbox
    out = []
    for f in ann[:k]:
        coords = f['geometry']['coordinates']
        # flatten a bit defensively
        arr = np.array(coords, dtype=float).reshape(-1, 2)
        xs, ys = arr[:,0], arr[:,1]
        props = f.get('properties', {})
        cls = props.get('classification', {})
        cname = cls.get('name', props.get('name', ''))
        out.append(dict(
            label=str(cname),
            n_pts=len(arr),
            minx=float(xs.min()), maxx=float(xs.max()),
            miny=float(ys.min()), maxy=float(ys.max()),
        ))
    return out

def _get(props, keys):
    for k in keys:
        if k in props:
            try:
                return float(props[k])
            except Exception:
                pass
    return None

def _scale_thickness_to_um(lst, mpp_x, mpp_y):
    # use mean if isotropic; replace with per-axis formula if needed
    mpp = 0.5 * (mpp_x + mpp_y)
    out = []
    for v in lst:
        if v is None:
            out.append(None)
        elif v >= 0:
            out.append(v * mpp)   # pixels -> micrometers
        else:
            out.append(v)         # keep sentinel negatives
    return out

def get_mpp_xy(slide_0):
    """
    Try common keys across SVS/NDPI.
    Returns (mpp_x, mpp_y) in microns/pixel.
    Raises if missing.
    """
    p = slide_0.properties

    # OpenSlide standard (most SVS/NDPI will have these):
    mpp_x = _get(p, ['openslide.mpp-x'])
    mpp_y = _get(p, ['openslide.mpp-y'])

    # Some vendors store a single isotropic MPP (e.g., Aperio):
    iso = _get(p, ['aperio.MPP', 'tiffslide.mpp'])
    if mpp_x is None and iso is not None:
        mpp_x = iso
    if mpp_y is None and iso is not None:
        mpp_y = iso

    # Fallbacks (rare/legacy vendor keys)
    # Note: if these are in *cm per pixel* or *dpi*, you’d need to convert.
    # The ones below are left commented unless you know their units.
    # mpp_x = mpp_x or _get(p, ['hamamatsu.XResolution'])
    # mpp_y = mpp_y or _get(p, ['hamamatsu.YResolution'])

    if mpp_x is None or mpp_y is None:
        raise ValueError("Could not find MPP in slide properties")

    return mpp_x, mpp_y

def get_label_safe(artery_id, wsi_id, df_label):
    """Return label or np.nan if the table or entry is missing."""
    if df_label is None:
        return np.nan
    try:
        return df_label.loc[artery_id, wsi_id]
    except Exception:
        return np.nan

def _label_of(feat):
    p = feat.get('properties', {}) or {}
    cls = p.get('classification') or {}
    return str(cls.get('name', p.get('name',''))).strip()

def _bbox_from_coords(coords):
    arr = np.array(coords, dtype=float).reshape(-1, 2)
    return (float(arr[:,0].min()), float(arr[:,0].max()),
            float(arr[:,1].min()), float(arr[:,1].max()))

def wsi_analysis(slide, ann, slide_boundries, wsi_id, df, um2_per_px, mpp_x, mpp_y):
    # Pick outers strictly by label in the same (global) frame as the rest
    outers = [f for f in ann if _label_of(f).lower() == 'media']
    print(f"Selected {len(outers)} outer polygons (label='Media') for {wsi_id}", flush=True)

    # Quick sanity prints
    sxmin, sxmax, symin, symax = slide_boundries
    for i, f in enumerate(outers[:5], 1):
        ob = _bbox_from_coords(f['geometry']['coordinates'])
        print(f"  outer[{i}] bbox x[{ob[0]:.1f},{ob[1]:.1f}] y[{ob[2]:.1f},{ob[3]:.1f}] "
              f"within global window x[{sxmin},{sxmax}] y[{symin},{symax}]", flush=True)

    for i, ann_outer in enumerate(outers):
        props = ann_outer.get('properties', {}) or {}
        # give it a readable id if none
        artery_id = props.get('name') or f"MediaOuter-{i+1}"
        print(artery_id, flush=True)
        # df = artery_analysis(slide, ann_outer, ann, slide_boundries, wsi_id, artery_id, df)
        df = artery_analysis(slide, ann_outer, ann, slide_boundries, wsi_id, artery_id, df, um2_per_px, mpp_x, mpp_y)
    return df

def artery_analysis(slide, ann_outer, ann, slide_boundries, wsi_id, artery_id, df, um2_per_px, mpp_x, mpp_y):

    # --- 0) Read polygons in GLOBAL frame (from file) ---
    cnt_outer_G = ann_outer["geometry"]["coordinates"]           # GLOBAL
    cnts_middle_G = get_cnts_inside(ann, cnt_outer_G, target="Intima")  # GLOBAL
    cnts_inner_G  = get_cnts_inside(ann, cnt_outer_G, target="Lumen")   # GLOBAL
    if len(cnts_inner_G) == 0 or len(cnts_middle_G) == 0:
        print(f"No Intima/Media Contour: {artery_id}", flush=True)
        return df

    # --- 1) Compute GLOBAL bbox of outer; do NOT shift to slide yet ---
    axmin, axmax, aymin, aymax = get_border_of_cnt(cnt_outer_G, border=50)

    sxmin, sxmax, symin, symax = slide_boundries
    H, W = slide.shape[:2]

    # Compute crop coords in SLIDE-LOCAL frame
    lxmin = int(max(0, axmin - sxmin))
    lymin = int(max(0, aymin - symin))
    lxmax = int(min(W, axmax - sxmin))
    lymax = int(min(H, aymax - symin))

    print(f"[{wsi_id}:{artery_id}] "
          f"bbox_global=({axmin},{axmax},{aymin},{aymax})  "
          f"slide_bbox=({sxmin},{sxmax},{symin},{symax})  "
          f"=> crop_local=({lxmin},{lxmax},{lymin},{lymax})  "
          f"slide_size={W}x{H}", flush=True)

    if lxmax <= lxmin or lymax <= lymin:
        print(f"Empty/invalid crop for {wsi_id} {artery_id} – skipping", flush=True)
        return df

    # --- 2) Crop the slide (this is SLIDE-LOCAL image) ---
    curr_slide = slide[lymin:lymax, lxmin:lxmax].copy()

    # --- 3) Shift GLOBAL polygons to CROP-LOCAL frame by subtracting (axmin, aymin) ---
    def _to_crop_local(poly_G):
        arr = np.asarray(poly_G, dtype=float).reshape(-1, 2)
        arr[:, 0] -= axmin
        arr[:, 1] -= aymin
        return arr.astype(np.int32)

    cnt_outer_L   = _to_crop_local(cnt_outer_G)
    cnts_middle_L = [_to_crop_local(c) for c in cnts_middle_G]
    cnts_inner_L  = [_to_crop_local(c) for c in cnts_inner_G]

    # Sanity prints: crop-local coordinates should be ~0..(lxmax-lxmin / lymax-lymin)
    ox0, ox1 = cnt_outer_L[:,0].min(), cnt_outer_L[:,0].max()
    oy0, oy1 = cnt_outer_L[:,1].min(), cnt_outer_L[:,1].max()
    print(f"  crop dims: {curr_slide.shape[1]}x{curr_slide.shape[0]}  "
          f"outer_local x[{ox0},{ox1}] y[{oy0},{oy1}]", flush=True)

    # --- 4) Draw overlays (OpenCV expects uint8) ---
    curr_slide = plot_artery_ann(curr_slide, cnt_outer_L, cnts_middle_L, cnts_inner_L)
    h, w = curr_slide.shape[:2]
    curr_ann = np.zeros((h, w, 3), np.uint8)
    curr_ann = plot_artery_ann(curr_ann, cnt_outer_L, cnts_middle_L, cnts_inner_L)

    # --- 5) Areas in local frame (translation-invariant anyway) ---
    # area_lumen  = float(np.sum([cv2.contourArea(c) for c in cnts_inner_L]))
    # area_intima = float(np.sum([cv2.contourArea(c) for c in cnts_middle_L]) - area_lumen)
    # area_media  = float(cv2.contourArea(cnt_outer_L) - area_intima - area_lumen)
    area_lumen_px  = float(np.sum([cv2.contourArea(c) for c in cnts_inner_L]))
    area_intima_px = float(np.sum([cv2.contourArea(c) for c in cnts_middle_L]) - area_lumen_px)
    area_media_px  = float(cv2.contourArea(cnt_outer_L) - area_intima_px - area_lumen_px)

    # Convert to µm²
    area_lumen_um2  = area_lumen_px  * um2_per_px
    area_intima_um2 = area_intima_px * um2_per_px
    area_media_um2  = area_media_px  * um2_per_px

    # --- 6) Thickness (guard against topologic edge cases inside your util) ---
    max_lumen = 0.0
    for i_in, cnt_in in enumerate(cnts_inner_L):
        for i_mid, cnt_mid in enumerate(cnts_middle_L):
            if not cnt_polygon_test(cnt_in, cnt_mid):
                continue

            exclude = (cnts_middle_L[:i_mid] + cnts_middle_L[i_mid+1:]
                       + cnts_inner_L[:i_in] + cnts_inner_L[i_in+1:])

            curr_area_intima = float(cv2.contourArea(cnt_mid))
            curr_area_lumen  = float(cv2.contourArea(cnt_in))
            if curr_area_lumen < max_lumen:
                continue
            max_lumen = curr_area_lumen

            thick_media, thick_intima = measure_thickness(
                cnt_outer_L, cnt_mid, cnt_in,
                wsi_id=wsi_id, artery_id=artery_id,
                angle_width=15, exclude=exclude,
                vis=curr_ann, dir_parent=DIR_SAVE_FIGURE
            )
            thick_media_um  = _scale_thickness_to_um(thick_media,  mpp_x, mpp_y)
            thick_intima_um = _scale_thickness_to_um(thick_intima, mpp_x, mpp_y)

            row = {
                'WSI_ID': wsi_id,
                'Artery_ID': artery_id,
                'MPP_X': mpp_x,
                'MPP_Y': mpp_y,
                'Thickness_Media': thick_media,
                'Thickness_Intima': thick_intima,
                'Thickness_Media_px': thick_media,
                'Thickness_Intima_px': thick_intima,
                'Thickness_Media_um': thick_media_um,
                'Thickness_Intima_um': thick_intima_um,
                'Curr_Area_Intima': curr_area_intima,
                'Curr_Area_Lumen': curr_area_lumen,
                'Area_Media':  area_media_px,
                'Area_Intima': area_intima_px,
                'Area_Lumen':  area_lumen_px,
                'Area_Media_um2': area_media_um2,
                'Area_Intima_um2': area_intima_um2,
                'Area_Lumen_um2': area_lumen_um2,
            }
            df.loc[len(df)] = row

    return df

# ---- feature extraction per row (requires your post_process & extract_features to be defined) ----
def compute_row_features(row):
    # m_raw = np.asarray(row["Thickness_Media"], dtype=float)
    # i_raw = np.asarray(row["Thickness_Intima"], dtype=float)
    # w_raw = np.where(m_raw >= 0, m_raw + i_raw, m_raw)
    m_raw = np.asarray(row["Thickness_Media_um"], dtype=float)
    i_raw = np.asarray(row["Thickness_Intima_um"], dtype=float)
    w_raw = np.where(m_raw >= 0, m_raw + i_raw, m_raw)
    m, i, r = post_process(m_raw, i_raw, w_raw)
    f_intima, f_media, f_ratio = extract_features(m, i, r)

    # µm² areas (new)
    am_um2 = float(row["Area_Media_um2"])
    ai_um2 = float(row["Area_Intima_um2"])
    al_um2 = float(row["Area_Lumen_um2"])
    at_um2 = am_um2 + ai_um2 + al_um2

    # Pixel areas (legacy) – may be missing, so use .get()
    am_px = row.get("Area_Media", np.nan)
    ai_px = row.get("Area_Intima", np.nan)
    al_px = row.get("Area_Lumen", np.nan)
    cai_px = row.get("Curr_Area_Intima", np.nan)  # you already have these in df
    cal_px = row.get("Curr_Area_Lumen", np.nan)

    area_feats = {
        # resolution
        "MPP_X": row["MPP_X"],
        "MPP_Y": row["MPP_Y"],
        # µm²
        "Media Area (microm2)":  am_um2,
        "Intima Area (microm2)": ai_um2,
        "Lumen Area (microm2)":  al_um2,
        "Media Area Frac":   (am_um2 / at_um2) if at_um2 > 0 else np.nan,
        "Intima Area Frac":  (ai_um2 / at_um2) if at_um2 > 0 else np.nan,
        "Lumen Area Frac":   (al_um2 / at_um2) if at_um2 > 0 else np.nan,
        "Ratio Intima/Media Area": (ai_um2 / am_um2) if am_um2 > 0 else np.nan,
        "Area_Media_px": am_px,
        "Area_Intima_px": ai_px,
        "Area_Lumen_px": al_px,
        "Curr_Area_Intima_px": cai_px,
        "Curr_Area_Lumen_px": cal_px,
        # "Thickness_Media_um": m_raw,
        # "Thickness_Intiam_um": i_raw
    }

    return {
        "WSI_ID": row["WSI_ID"],
        "Artery_ID": row["Artery_ID"],
        "WSI_Artery_ID": row["WSI_ID"] + "__" + row["Artery_ID"],
        **area_feats, **f_intima, **f_media, **f_ratio,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Artery wall thickness and feature extraction pipeline")
    parser.add_argument("--wsi_dir", required=True, help="Path to directory containing WSI files (.svs or .ndpi)")
    parser.add_argument("--ann_dir", required=True, help="Path to directory containing corresponding .geojson annotations")
    parser.add_argument("--save_dir", required=True, help="Directory to save results and figures")
    args = parser.parse_args()

    # --- Directories ---
    DIR_WSI = args.wsi_dir
    DIR_ANN = args.ann_dir
    DIR_SAVE_RESULTS = args.save_dir
    DIR_SAVE_FIGURE = os.path.join(DIR_SAVE_RESULTS, "figures")
    os.makedirs(DIR_SAVE_RESULTS, exist_ok=True)
    os.makedirs(DIR_SAVE_FIGURE, exist_ok=True)

    # --- Init ---
    start_time = time.time()
    df = pd.DataFrame(columns=[
        'WSI_ID','Artery_ID','MPP_X','MPP_Y',
        'Thickness_Media','Thickness_Intima',
        'Thickness_Media_um','Thickness_Intima_um',   
        'Curr_Area_Intima','Curr_Area_Lumen',
        'Area_Media','Area_Intima','Area_Lumen',
        'Area_Media_um2','Area_Intima_um2','Area_Lumen_um2'
    ])

    # --- Process each WSI ---
    for wsi in os.listdir(DIR_WSI):
        if not (wsi.endswith(".ndpi") or wsi.endswith(".svs")):
            continue

        start_time_wsi = time.time()
        path_wsi = os.path.join(DIR_WSI, wsi)
        wsi_id = ".".join(wsi.split(".")[:-1])
        print(f"Working on Case: {wsi_id}")

        path_ann = os.path.join(DIR_ANN, wsi_id + ".geojson")
        if not os.path.exists(path_ann):
            print(f" Missing annotation file: {path_ann}")
            continue

        with open(path_ann) as f:
            exported = geojson.load(f)
            ann = exported["features"]

        for i in range(len(ann)):
            coords_raw = ann[i]["geometry"]["coordinates"]
            ann[i]["geometry"]["coordinates"] = clean_coord(coords_raw)

        print("Label histogram:", _hist_labels(ann))
        print("Sample polygons:", _first_polys_info(ann, k=5))

        xmin, xmax, ymin, ymax = get_border_of_ann(ann)
        print(f"[{wsi_id}] global window x[{xmin},{xmax}] y[{ymin},{ymax}] size=({xmax-xmin},{ymax-ymin})")

        slide_0 = openslide.OpenSlide(path_wsi)
        mpp_x, mpp_y = get_mpp_xy(slide_0)
        um2_per_px = mpp_x * mpp_y

        slide = slide_0.read_region((xmin, ymin), 0, (xmax - xmin, ymax - ymin))
        slide = np.asarray(slide)
        slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)

        df = wsi_analysis(slide, ann, (xmin, xmax, ymin, ymax), wsi_id, df, um2_per_px, mpp_x, mpp_y)
        print(f"{wsi_id} done in {time.time() - start_time_wsi:.2f}s")

    print(f"Total runtime: {time.time() - start_time:.2f}s")

    # --- Save JSON + CSV outputs ---
    json_path = os.path.join(DIR_SAVE_RESULTS, "thickness.json")
    df.to_json(json_path, orient="records", lines=True)
    print(f" Saved thickness measurements → {json_path}")

    # Merge with legacy if available
    if os.path.exists(json_path):
        df_json = pd.read_json(json_path, lines=True)
        keep_cols = ["WSI_ID", "Artery_ID", "Area_Media", "Area_Intima", "Area_Lumen", "Curr_Area_Intima", "Curr_Area_Lumen"]
        df_json = df_json[[c for c in keep_cols if c in df_json.columns]]
        df = df.merge(df_json, on=["WSI_ID", "Artery_ID"], how="left", suffixes=("", "_legacy"))

    df["WSI_Artery_ID"] = df["WSI_ID"] + "__" + df["Artery_ID"]
    features = [compute_row_features(r) for _, r in df.iterrows()]
    df_features = pd.DataFrame(features)
    out_csv = os.path.join(DIR_SAVE_RESULTS, "artery_features_no_labels.csv")
    df_features.to_csv(out_csv, index=False)
    print(f" Saved final features → {out_csv}")