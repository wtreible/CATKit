# Default imports
import os, sys, pdb

# ML and CV library imports
import cv2
import torch
import numpy as np

# Import CATLib packages
sys.path.append('/sdk/catlib')
from cat_db import CatDB
from cat_datasets import SegmentationDataset
from cat_utils import pprint, mdprint

# Set important paths and defaults
SCRATCH_BASEPATH = '/sdk/scratch/stereo/examples'
LEFT, RIGHT, DISP = 0, 1, 2
os.makedirs(SCRATCH_BASEPATH, mode=0o777, exist_ok=True)

# Initialize CatDB
catdb = CatDB()

# Get Image IDs
image_id_triples = []

for i in range(10):
  arrangement_str = 'arrangement' + str(i+1)
  left_img_id = catdb.multisearch('location','indoor',
                                  'scene', 'books',
                                  'arrangement', arrangement_str,
                                  'modality', 'cross',
                                  'type', 'rect',
                                  'hazard', 'default',
                                  'side', 'left')
  right_img_id = catdb.multisearch('location','indoor',
                                   'scene', 'books',
                                   'arrangement', arrangement_str,
                                   'modality', 'cross',
                                   'type', 'rect',
                                   'hazard', 'default',
                                   'side', 'right')
  disp_id = catdb.multisearch('location','indoor',
                              'scene', 'books',
                              'arrangement', arrangement_str,
                              'modality', 'cross',
                              'type', 'disparity')
  image_id_triples.append(left_img_id + right_img_id + disp_id)

# Verify item names are correct
for temp_triple in image_id_triples:
  left_item = catdb[temp_triple[LEFT]]
  right_item = catdb[temp_triple[RIGHT]]
  disp_item = catdb[temp_triple[DISP]]
  mdprint(left_item, right_item, disp_item, keys=['name','arrangement'])