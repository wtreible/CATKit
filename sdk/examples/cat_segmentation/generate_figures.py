# Default imports
import os, sys, pdb, glob

# ML and CV library imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create paths
LEGEND_FILE_PATH = 'legend/legend.png'
SCRATCH_BASEPATH = '/sdk/scratch/cat_segmentation/examples'
FIGURES_PATH = os.path.join(SCRATCH_BASEPATH, 'Figures')
try:
  os.makedirs(FIGURES_PATH)
except:
  pass
  
# Set Default Variables
EDGE_IMGS = 'EdgeNet'
NETWORKS = ['UNet', 'RTFNet', 'WHERECAT']
IDS = [1, 12, 13, 19, 25, 31, 32, 40, 44]

# Figure Defaults
TEXT_HEIGHT = 50
IMG_SIZE = (433,325)

# Create Figures
legend = cv2.imread(LEGEND_FILE_PATH) # h:650, w:400
for network in NETWORKS:
  network_dir = os.path.join(SCRATCH_BASEPATH, network)
  for id in IDS:
    gt_img_path = glob.glob(network_dir + '/gt_mask_*_{}.png'.format(id))[0]
    pred_img_path = glob.glob(network_dir + '/pred_mask_*_{}.png'.format(id))[0]
    thermal_img_path = glob.glob(network_dir + '/thermal_image_*_{}.png'.format(id))[0]
    thermal_pred_path = glob.glob(network_dir + '/pred_thermal_mask_*_{}.png'.format(id))[0]
    gt_img = cv2.imread(gt_img_path)
    pred_img = cv2.imread(pred_img_path)
    thermal_img = cv2.imread(thermal_img_path)
    thermal_pred = cv2.imread(thermal_pred_path)
    gt_img = cv2.resize(gt_img, (433,325))
    pred_img = cv2.resize(pred_img, (433,325))
    thermal_img = cv2.resize(thermal_img, (433,325))
    thermal_pred = cv2.resize(thermal_pred, (433,325))
    
    legend_y = legend.shape[0]
    legend_x = legend.shape[1]
    img_y = gt_img.shape[0]
    img_x = gt_img.shape[1]
    figure_out = np.zeros((legend_y, legend_x + 2*img_x, 3))
    figure_out[:legend_y, :legend_x] = legend
    figure_out[:img_y,  legend_x:legend_x+img_x] = gt_img
    figure_out[img_y:,  legend_x:legend_x+img_x] = pred_img
    figure_out[:img_y,  legend_x+img_x:legend_x+2*img_x] = thermal_img
    figure_out[img_y:,  legend_x+img_x:legend_x+2*img_x] = thermal_pred
    cv2.imwrite(os.path.join(FIGURES_PATH,'ex{}_{}.png'.format(id,network)), figure_out)