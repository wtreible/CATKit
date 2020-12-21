# Default imports
import os, sys, pdb

# ML and CV library imports
import cv2
import torch
import numpy as np
from skimage.color import label2rgb

# External imports
import albumentations as albu
import segmentation_models_pytorch as smp

# Import CATLib packages
sys.path.append('/sdk/catlib')
from cat_db import CatDB
from cat_datasets import SegmentationDataset
from cat_utils import read_label_names, get_unique_classes, img_float_to_uint8

# Set important paths and defaults
EPOCHS = 40
ENCODERS = ['se_resnext50_32x4d', 'densenet161', 'resnet50', 'efficientnet-b5', 'timm-efficientnet-b5	']
MODELS = ['unet
ENCODER = 'se_resnext50_32x4d'
BEST_MODEL_PATH = 'best_model_' + ENCODER + '_imagenet.pth'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
SCRATCH_BASEPATH = '/sdk/scratch/segmentation/examples'
os.makedirs(SCRATCH_BASEPATH, mode=0o777, exist_ok=True)

unique_classes = ['Book', 'Cloth', 'Container', 'Cork', 'Lamp', 'Metal', 'Pen', 'Plant', 'Rock', 'Statue', 'Styrofoam', 'Wood']

# If a colors file was supplied, load it from disk
colors_file_path = 'colors.csv'
CLASSES = ['Background'] + unique_classes
if os.path.isfile(colors_file_path):
  COLORS = open(colors_file_path).read().strip().split("\n")
  COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
  COLORS = np.array(COLORS, dtype="uint8")
# otherwise, we need to randomly generate RGB colors for each class
# label
else:
  np.random.seed(42)
  COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
    dtype="uint8")
  COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

COLORS = COLORS[:len(CLASSES)]

# initialize the legend visualization
WIDTH = 400
TEXT_HEIGHT = 50
legend = np.zeros(((len(CLASSES[1:]) * TEXT_HEIGHT), WIDTH, 3), dtype="uint8")
# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES[1:], COLORS[1:])):
  # draw the class name + color on the legend
  color = [int(c) for c in color]
  cv2.putText(legend, className, (TEXT_HEIGHT//5, (i * TEXT_HEIGHT) + 32),
    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
  cv2.rectangle(legend, (200, (i * TEXT_HEIGHT)), (WIDTH, (i * TEXT_HEIGHT) + TEXT_HEIGHT),
    tuple(color), -1)
cv2.imwrite('legend.png', legend)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Search for item ids from the CatDB
#  For this example, only use "books" scene with the (raw, left, color, default)
#  images that correspond to the label images
catdb = CatDB()

label_img_ids = catdb.multisearch('location','indoor',
                                  'type', 'labels',
                                  'label_names','!-',
                                  'arrangement','arrangement1')
label_img_ids += catdb.multisearch('location','indoor',
                                  'type', 'labels',
                                  'label_names','!-',
                                  'arrangement','arrangement2')
raw_img_ids = []
for l_id in label_img_ids:
  tmp_item = catdb[l_id]
  raw_img_id = catdb.multisearch('arrangement',tmp_item['arrangement'],
                                 'scene',tmp_item['scene'], 
                                 'type', 'raw', 
                                 'side', 'left', 
                                 'hazard', 'default', 
                                 'modality', 'color')
  raw_img_ids.append(raw_img_id[0])

seg_dataset = SegmentationDataset(catdb, raw_img_ids, label_img_ids, 
                                  valid_classes=unique_classes,
                                  preprocessing_fn=preprocessing_fn)

# load best saved checkpoint and test image
best_model_path = os.path.join(SCRATCH_BASEPATH, BEST_MODEL_PATH)
best_model = torch.load(best_model_path)
for i in range(len(seg_dataset)):
  test_image, test_mask = seg_dataset[i]
  gt_mask = np.transpose(test_mask, (1,2,0))
  # Convert test image to a tensor
  test_tensor = torch.from_numpy(np.expand_dims(test_image,0)).float().to(DEVICE)

  # Predict mask
  output = best_model.predict(test_tensor)

  # Save output segmentation image
  image = np.transpose(test_image, (1,2,0))
  mask = np.transpose(np.squeeze(output.cpu().numpy()), (1,2,0))
  image = img_float_to_uint8(image)
  intensity_mask = np.argmax(mask, axis=-1)
  
  # Calculate IoU
  target = gt_mask[...,1:]
  prediction = np.rint(mask[...,1:])
  intersection = np.logical_and(target, prediction)
  union = np.logical_or(target, prediction)
  iou_score = np.sum(intersection) / np.sum(union)
  
  out_img = np.array(image)
  for val, clr in enumerate(COLORS):
    if val != 0:
      out_img[intensity_mask == val] = clr * 0.85 + out_img[intensity_mask == val] * 0.15
  
  with open(os.path.join(SCRATCH_BASEPATH,'output_iou{}.txt'.format(i)), 'w') as fp:
    fp.write(str(iou_score))
  cv2.imwrite(os.path.join(SCRATCH_BASEPATH,'output_img{}.png'.format(i)), image.astype(np.uint8))
  cv2.imwrite(os.path.join(SCRATCH_BASEPATH,'output_mask{}.png'.format(i)), out_img.astype(np.uint8))