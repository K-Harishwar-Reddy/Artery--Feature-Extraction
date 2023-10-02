import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

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