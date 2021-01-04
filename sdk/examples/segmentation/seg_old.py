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
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
SCRATCH_BASEPATH = '/sdk/scratch/segmentation/examples'
os.makedirs(SCRATCH_BASEPATH, mode=0o777, exist_ok=True)

# Search for item ids from the CatDB
#  For this example, only use "books" scene with the (raw, left, color, default)
#  images that correspond to the label images
catdb = CatDB()
raw_img_ids = catdb.multisearch('scene', 'books', 'type', 'raw', 'side', 'left', 'hazard', 'default', 'modality', 'color')
label_img_ids = catdb.multisearch('scene', 'books', 'type', 'labels')
assert(len(raw_img_ids) == len(label_img_ids))

# Determine unique classes in the data
label_items = catdb[label_img_ids]
label_names_list = read_label_names(label_items)
unique_classes, unique_subclasses = get_unique_classes(label_names_list)
NUM_CLASSES = len(unique_classes) + 1 # Add in background
print ("Unique Classes in the dataset: {}".format(unique_classes))

# Build a pretrained model for finetuning
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=NUM_CLASSES, 
    activation=ACTIVATION,
)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Create a SegmentationDataset & DataLoader
seg_dataset = SegmentationDataset(catdb, raw_img_ids, label_img_ids, 
                                  valid_classes=unique_classes,
                                  preprocessing_fn=preprocessing_fn)
train_loader = torch.utils.data.DataLoader(seg_dataset, batch_size=1, shuffle=False, num_workers=1)

# Save some images to verify augmentation is working
for i in range(3):
  image, mask = seg_dataset[i]
  image = np.transpose(image, (1,2,0))
  mask = np.transpose(mask, (1,2,0))
  image = img_float_to_uint8(image)
  intensity_mask = np.argmax(mask, axis=-1)
  rgb_mask = label2rgb(intensity_mask, image=image, bg_label=0)
  rgb_mask = img_float_to_uint8(rgb_mask)
  cv2.imwrite(os.path.join(SCRATCH_BASEPATH,'label_example{}.png'.format(i)), rgb_mask)


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

# Train Model
best_model_path = ''
max_score = 0
for epoch in range(0, EPOCHS):
  print('Epoch: {}'.format(epoch))
  train_logs = train_epoch.run(train_loader)
  if max_score < train_logs['iou_score']:
    max_score = train_logs['iou_score']
    model_name = 'best_model.pth' #'best_model_IoU-{}_E-{}.pth'.format(max_score, epoch)
    best_model_path = os.path.join(SCRATCH_BASEPATH, model_name)
    torch.save(model, best_model_path)
    print('Model saved!')
      
  if epoch == 25:
    optimizer.param_groups[0]['lr'] = 1e-5
    print('Changing Learning Rate...')


# load best saved checkpoint and test image
best_model = torch.load(best_model_path)
test_image, test_mask = seg_dataset[0]

# Convert test image to a tensor
test_tensor = torch.from_numpy(np.expand_dims(test_image,0)).float().to(DEVICE)

# Predict mask
output = best_model.predict(test_tensor)

# Save output segmentation image
image = np.transpose(test_image, (1,2,0))
mask = np.transpose(np.squeeze(output.cpu().numpy()), (1,2,0))
image = img_float_to_uint8(image)
intensity_mask = np.argmax(mask, axis=-1)
rgb_mask = label2rgb(intensity_mask, image=image, bg_label=0)
rgb_mask = img_float_to_uint8(rgb_mask)
cv2.imwrite(os.path.join(SCRATCH_BASEPATH,'output_test.png'), rgb_mask)

pdb.set_trace()
