# %%
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
import pydicom

# %%
DCM_DIR = r'J:\ML_for_porfolio\segmentation\unet_rsna\data\stage_2_train_images'
CSV_DIR = r'J:\ML_for_porfolio\segmentation\unet_rsna\data\stage_2_train_labels.csv'
REAL_IMAGES_DIR = r'J:\ML_for_porfolio\segmentation\unet_rsna\data\images'
MASKS_DIR = r'J:\ML_for_porfolio\segmentation\unet_rsna\data\masks'

# %%
info = pd.read_csv(CSV_DIR)

info = info[info.Target == 1]

# %%
unique_patients = info.patientId.nunique()
total_bboxes = len(info)

print(f'Count of unique patients: {unique_patients} | Count of total bboxes: {total_bboxes}')

# %%
info = info.groupby(info.patientId)

# %%
for patient_id, group in tqdm(info):
    dcm_path = os.path.join(DCM_DIR,f'{patient_id}.dcm')
    
    if not os.path.exists(dcm_path):
        print(f"This file doesnt exsit: {dcm_path}")
        continue
    dicom = pydicom.dcmread(dcm_path)
    img = dicom.pixel_array
    
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(os.path.join(REAL_IMAGES_DIR, f"{patient_id}.png"), img_color)
    
    mask  = np.zeros_like(img,dtype=np.uint8)
    
    for row in group.iterrows():
        x, y, width, height = row[1][1:5].astype(int)
        
        top_right_x = x
        top_right_y = y
        bottom_left_x = x + width
        bottom_left_y = y + height
        
        cv2.rectangle(mask,(top_right_x,top_right_y),(bottom_left_x,bottom_left_y),color=255,thickness=-1)
        
    cv2.imwrite(os.path.join(MASKS_DIR, f"{patient_id}.png"), mask)
