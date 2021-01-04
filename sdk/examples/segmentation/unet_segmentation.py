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
ENCODERS = ['se_resnext50_32x4d', 'densenet161', 'resnet50', 'efficientnet-b5', 'timm-efficientnet-b5']
MODELS = ['PSPNet'] #'FPN', 'Unet', 'Linknet', 'DeepLabV3Plus']
COLORS_FILE_PATH = 'colors.csv'
LEGEND_FILE_PATH = 'legend.png'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cpu'
SCRATCH_BASEPATH = '/sdk/scratch/segmentation/examples'
os.makedirs(SCRATCH_BASEPATH, mode=0o777, exist_ok=True)

# Initialize CatDB
catdb = CatDB()

# Get Training Image IDs
label_img_ids = catdb.multisearch('location','indoor',
                                  'type', 'labels',
                                  'label_names','!-',
                                  'arrangement','!arrangement1',
                                  'arrangement','!arrangement2')
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

# Get Validation Image IDs
val_label_img_ids = catdb.multisearch('location','indoor',
                                  'type', 'labels',
                                  'label_names','!-',
                                  'arrangement','arrangement2')
val_raw_img_ids = []
for l_id in val_label_img_ids:
  tmp_item = catdb[l_id]
  raw_img_id = catdb.multisearch('arrangement',tmp_item['arrangement'],
                                 'scene',tmp_item['scene'], 
                                 'type', 'raw', 
                                 'side', 'left', 
                                 'hazard', 'default', 
                                 'modality', 'color')
  val_raw_img_ids.append(raw_img_id[0])

# Get Test Image IDs
test_label_img_ids = catdb.multisearch('location','indoor',
                                  'type', 'labels',
                                  'label_names','!-',
                                  'arrangement','arrangement1')
test_raw_img_ids = []
for l_id in test_label_img_ids:
  tmp_item = catdb[l_id]
  raw_img_id = catdb.multisearch('arrangement',tmp_item['arrangement'],
                                 'scene',tmp_item['scene'], 
                                 'type', 'raw', 
                                 'side', 'left', 
                                 'hazard', 'default', 
                                 'modality', 'color')
  test_raw_img_ids.append(raw_img_id[0])

# Determine unique classes in the data
label_items = catdb[label_img_ids]
label_names_list = read_label_names(label_items)
unique_classes, unique_subclasses = get_unique_classes(label_names_list)
print ("Unique Classes in the dataset: {}".format(unique_classes))

# Load Colors file
CLASSES = ['Background'] + list(unique_classes)
if not os.path.isfile(COLORS_FILE_PATH):
  sys.exit('No colors file provided!')
COLORS = open(COLORS_FILE_PATH).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")
COLORS = COLORS[:len(CLASSES)]

# Create legend visualization if it doesn't exist
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
cv2.imwrite(LEGEND_FILE_PATH, legend)

# Iterate over all models and encoders
for MODEL_NAME in MODELS:
  for ENCODER in ENCODERS:
  
    # Build a pretrained model for finetuning
    MODEL = getattr(smp, MODEL_NAME)
    model = MODEL(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Create a SegmentationDataset & DataLoader
    seg_dataset = SegmentationDataset(catdb, raw_img_ids, label_img_ids, 
                                      valid_classes=unique_classes,
                                      preprocessing_fn=preprocessing_fn)
    val_dataset = SegmentationDataset(catdb, val_raw_img_ids, val_label_img_ids, 
                                      valid_classes=unique_classes,
                                      preprocessing_fn=preprocessing_fn)                                  
    train_loader = torch.utils.data.DataLoader(seg_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Set loss function, metrics, and optimizer
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    # Create SMP training epoch runner
    train_epoch = smp.utils.train.TrainEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      optimizer=optimizer,
      device=DEVICE,
      verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      device=DEVICE,
      verbose=True,
    )

    # Train Model
    best_model_path = ''
    output_name = '{}_{}_{}'.format(MODEL_NAME, ENCODER, ENCODER_WEIGHTS)
    model_name = 'best_model_{}.pth'.format(output_name)
    os.makedirs(os.path.join(SCRATCH_BASEPATH, output_name), mode=0o777, exist_ok=True)
    max_score = 0
    '''
    for epoch in range(0, EPOCHS):
      print('Epoch: {}'.format(epoch))
      train_logs = train_epoch.run(train_loader)
      valid_logs = valid_epoch.run(valid_loader)
      
      if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        
        best_model_path = os.path.join(SCRATCH_BASEPATH, output_name, model_name)
        torch.save(model, best_model_path)
        print('Model saved!')
          
      if epoch == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Changing Learning Rate...')
    '''
    # TESTING
    print ("Running Tests...")
    test_dataset = SegmentationDataset(catdb, test_raw_img_ids, test_label_img_ids, 
                                       valid_classes=unique_classes,
                                       preprocessing_fn=preprocessing_fn)
                                       
    #best_model = torch.load(best_model_path)
    for i in range(len(test_dataset)):
      test_image, test_mask = test_dataset[i]
      gt_mask = np.transpose(test_mask, (1,2,0))
      # Convert test image to a tensor
      test_tensor = torch.from_numpy(np.expand_dims(test_image,0)).float().to(DEVICE)

      # Predict mask
      #output = best_model.predict(test_tensor)

      # Save output segmentation image
      #image = np.transpose(test_image, (1,2,0))
      #mask = np.transpose(np.squeeze(output.cpu().numpy()), (1,2,0))
      #image = img_float_to_uint8(image)
      #intensity_mask = np.argmax(mask, axis=-1)
      
      # Calculate IoU
      '''
      target = gt_mask[...,1:]
      prediction = np.rint(mask[...,1:])
      intersection = np.logical_and(target, prediction)
      union = np.logical_or(target, prediction)
      iou_score = np.sum(intersection) / np.sum(union)
      '''

      intensity_mask = np.argmax(gt_mask, axis=-1)
      out_img = np.zeros(intensity_mask.shape + (3,))
      for val, clr in enumerate(COLORS):
        if val != 0:
          out_img[intensity_mask == val] = clr # * 0.85 + out_img[intensity_mask == val] * 0.15
          
      cv2.imwrite('gt_mask{}.png'.format(i), out_img.astype(np.uint8))
      '''
      with open(os.path.join(SCRATCH_BASEPATH, output_name, 'output_iou{}.txt'.format(i)), 'w') as fp:
        fp.write(str(iou_score))
      cv2.imwrite(os.path.join(SCRATCH_BASEPATH, output_name, 'output_img{}.png'.format(i)), image.astype(np.uint8))
      cv2.imwrite(os.path.join(SCRATCH_BASEPATH,output_name, 'output_mask{}.png'.format(i)), out_img.astype(np.uint8))
      '''
      pdb.set_trace()
    print ("Done with tests!")
    