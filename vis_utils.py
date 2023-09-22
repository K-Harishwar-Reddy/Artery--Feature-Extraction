import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
DIR_SAVE_FIGURE = "/Users/jinzhou/Desktop/USCAP/results/figures"

def plot_artery_ann(vis, cnt_outer, cnts_mid, cnts_inner, cnt_thick=2):
    cv2.drawContours(vis, [cnt_outer], -1, [255, 0, 0], cnt_thick)
    cv2.drawContours(vis, cnts_mid, -1, [0, 255, 0], cnt_thick)
    cv2.drawContours(vis, cnts_inner, -1, [0, 0, 255], cnt_thick)
    return vis
    
def save_img(img, parent_dir, wsi_id, artery_id):
    dir_save_wsi = os.path.join(parent_dir, wsi_id)
    if not os.path.exists(dir_save_wsi):
        os.makedirs(dir_save_wsi)
    path_to_svae = os.path.join(dir_save_wsi, artery_id+'.png')
    Image.fromarray(img).save(path_to_svae)
    
def save_img_for_seg(img, parent_dir, wsi_id, artery_id, category):
    dir_save = os.path.join(parent_dir, category)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    path_to_svae = os.path.join(dir_save, wsi_id+'_'+artery_id+'.png')
    Image.fromarray(img).save(path_to_svae)

    
def save_img_animation_helper(img, dir_save_wsi, sample_id):
            
    if not os.path.exists(dir_save_wsi):
        os.makedirs(dir_save_wsi)
    if sample_id >= 10:
        sample_id_str = str(sample_id)
    else:
        sample_id_str = "0" + str(sample_id)
    path_to_svae = os.path.join(dir_save_wsi, sample_id_str+'.png')
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