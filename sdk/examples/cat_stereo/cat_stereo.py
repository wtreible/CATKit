# Default imports
import os, sys, pdb

# ML and CV library imports
import cv2
import torch
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import rescale
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

# Import CATLib packages
sys.path.append('/sdk/catlib')
from cat_db import CatDB
from cat_utils import pprint, mdprint

# Import Local Packages
from wildcat_model import WILDCAT
from cat_sgm import (read_images, WILDCAT_costvolume, aggregate_costs, 
                    select_disparity, get_recall, save_costvolume_image,
                    compute_costs, CATSGMParameters, SGMPaths)

# Set scratch path
SCRATCH_BASEPATH = '/sdk/scratch/cat_stereo/examples'
os.makedirs(SCRATCH_BASEPATH, mode=0o777, exist_ok=True)

# Initialize CatDB
catdb = CatDB()
metadb = catdb.get_metadb()

# Set defaults
PATCH_SIZE = (64, 64)
WEIGHTS_PATH = 'WILDCAT_RES_WEIGHTS.h5'
LEFT, RIGHT, DISP, MASK = 0, 1, 2, 3
TEST_SCENES = ['books'] #metadb['indoor_scene'] #['misc', 'materials']
TRAIN_SCENES = [s for s in metadb['indoor_scene'] if s not in TEST_SCENES]
ARRANGEMENTS = ['arrangement1', 'arrangement2'] #metadb['arrangement']
MODALITY = 'cross'

# Get Image ID Sets
def populate_id_sets(catdb, scenes, arrangements, modality):
  image_id_sets = []
  for scene in scenes:
    for arrangement in arrangements:
      left_img_id = catdb.multisearch('scene', scene,
                                      'arrangement', arrangement,
                                      'modality', modality,
                                      'type', 'rect',
                                      'hazard', 'default',
                                      'side', 'left')
      right_img_id = catdb.multisearch('scene', scene,
                                       'arrangement', arrangement,
                                       'modality', modality,
                                       'type', 'rect',
                                       'hazard', 'default',
                                       'side', 'right')
      mask_img_id = catdb.multisearch('scene', scene,
                                      'arrangement', arrangement,
                                      'modality', modality,
                                      'type', 'mask')
      disp_id = catdb.multisearch('scene', scene,
                                  'arrangement', arrangement,
                                  'modality', modality,
                                  'type', 'disparity')
      image_id_sets.append(left_img_id + right_img_id + disp_id + mask_img_id)
  return image_id_sets

def imadjust(img, low=1, high=99):
  plow, phigh = np.percentile(img, (low, high))
  return rescale_intensity(img, in_range=(plow, phigh))

def remove_zeros_imadjust(img, low=1, high=99):
  tmp = np.array(img)
  tmp[tmp == 0] = img.mean()
  plow, phigh = np.percentile(tmp, (low, high))
  return rescale_intensity(img, in_range=(plow, phigh))

TRAIN_ID_SET = populate_id_sets(catdb, TRAIN_SCENES, ARRANGEMENTS, MODALITY)
TEST_ID_SET = populate_id_sets(catdb, TEST_SCENES, ARRANGEMENTS, MODALITY)

# Initial Print
intro_str = "CAT Stereo Reconstruction Example"
print ('=' * len(intro_str))
print (intro_str)
print ('=' * len(intro_str))

# Load WILDCAT and dry run to prep cublas
MODEL = WILDCAT(PATCH_SIZE)
MODEL.load_weights(WEIGHTS_PATH)
MODEL.predict(np.random.rand(1,2,PATCH_SIZE[0],PATCH_SIZE[1]))

DISP_PAD = 64
PARAMETERS = CATSGMParameters(min_disparity=-32, max_disparity=64, P1=7, P2=49, csize=(7,7), wsize=PATCH_SIZE, bsize=(3, 3))
PATHS = SGMPaths()

output_path = os.path.join(SCRATCH_BASEPATH,'output')
try: 
  os.makedirs(output_path) 
except: 
  pass

# Verify testing set is correct
print ('Testing Image Sets:')
data = []
for i, temp_id_set in enumerate(TEST_ID_SET):
  
  left_item = catdb[temp_id_set[LEFT]]
  right_item = catdb[temp_id_set[RIGHT]]
  disp_item = catdb[temp_id_set[DISP]]
  mask_item = catdb[temp_id_set[MASK]]
  scene = left_item['scene']
  arrangement = left_item['arrangement']
  mdprint(left_item, right_item, disp_item, mask_item, keys=['scene', 'name','arrangement'], item_start='  ', item_end='\n')
  
  # Load images
  left, right, mask = read_images(left_item['path'], right_item['path'], mask_item['path'])
  
  # imadjust images
  left_imadj = remove_zeros_imadjust(left)
  right_imadj = remove_zeros_imadjust(right)
  
  left_imadj = rescale(left_imadj, 0.25, anti_aliasing=True)
  right_imadj = rescale(left_imadj, 0.25, anti_aliasing=True)
  
  # Load disparity
  d_interp = np.loadtxt(disp_item['interp_path'], delimiter=',')

  
  # Calculate Cost Volume
  left_cost_volume = WILDCAT_costvolume(MODEL, left_imadj, right_imadj, PARAMETERS, xstep=1, ystep=1, dstep=1)
  cv = np.argmin(left_cost_volume, axis=2)
  cv4x = rescale(cv, 4, anti_aliasing=True)
  #cost_volume = zoom(left_cost_volume, (ystep, xstep, dstep ))
  
  # 8-path Cost aggregation
  left_aggregation_volume = aggregate_costs(left_cost_volume, PARAMETERS, PATHS)
  
  # Select minimum costs for final disparity
  final_disparity = select_disparity(left_aggregation_volume)

  fpath = os.path.join(output_path,'{}_{}'.format(scene, arrangement))
  try: 
    os.makedirs(fpath) 
  except: 
    pass
  cv2.imwrite(os.path.join(fpath,'left.png'), left*255)
  cv2.imwrite(os.path.join(fpath,'right.png'), right*255)
  cv2.imwrite(os.path.join(fpath,'left_adj.png'), left_imadj*255)
  cv2.imwrite(os.path.join(fpath,'right_adj.png'), right_imadj*255)
  cv2.imwrite(os.path.join(fpath,'disp.tiff'), final_disparity.astype(np.float32)+PARAMETERS.min_disparity)
  cv2.imwrite(os.path.join(fpath,'gt_disp.tiff'), d_interp.astype(np.float32))
  cv2.imwrite(os.path.join(fpath,'cv_img.tiff'), cv)