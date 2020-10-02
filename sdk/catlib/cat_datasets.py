import pdb
import cv2
import numpy as np
import torch
import albumentations as albu
from cat_utils import get_label_name_maps

class SegmentationDataset(torch.utils.data.Dataset):
  'Segmentation Dataset for basic segmentation example'
  def __init__(self, catdb, img_ids, label_ids, valid_classes=None, class_map=None, use_subclasses=True, preprocessing_fn=None):
    self.catdb = catdb
    self.img_ids = img_ids
    self.label_ids = label_ids
    self.valid_classes = valid_classes
    self.use_subclasses = use_subclasses
    self.class_map = class_map
    
    self.augmentation = self._default_augmentation_fn()
    self.preprocessing = self._default_preprocessing_fn(preprocessing_fn)
      
  def _default_preprocessing_fn(self, preprocessing_fn):
    def to_tensor(x, **kwargs):
      return x.transpose(2, 0, 1).astype('float32')
    _transform = [
      albu.Lambda(image=preprocessing_fn),
      albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    
  def _default_augmentation_fn(self):
    train_transform = [
      albu.HorizontalFlip(p=0.5),
      albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
      albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
      albu.RandomCrop(height=512, width=512, always_apply=True),
      albu.IAAAdditiveGaussianNoise(p=0.2),
      albu.IAAPerspective(p=0.5),
      albu.OneOf([
          albu.CLAHE(p=1),
          albu.RandomBrightness(p=1),
          albu.RandomGamma(p=1),
        ],
        p=0.9,
      ),
      albu.OneOf([
          albu.IAASharpen(p=1),
          albu.Blur(blur_limit=3, p=1),
          albu.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
      ),
      albu.OneOf([
          albu.RandomContrast(p=1),
          albu.HueSaturationValue(p=1),
        ],
        p=0.9,
      ),
    ]
    return albu.Compose(train_transform)
    
  def __len__(self):
    return len(self.img_ids)

  def __getitem__(self, index):
    image_id = self.img_ids[index]
    label_id = self.label_ids[index]
    
    # Get the corresponding items from the CatDB
    image_item = self.catdb[image_id]
    label_item = self.catdb[label_id]
    
    # Read Images
    image = cv2.imread(image_item['path'])
    mask = cv2.imread(label_item['path'], 0)

    # Get classes that map to image intensity values
    classes_intensity_map, subclasses_intensity_map = get_label_name_maps(label_item)
    
    # If a class map exists... TO DO
    if self.class_map is not None:
      intensity_map = {}
      for classname in self.valid_classes:
        try:
          if self.use_subclasses:
            label_intensity_vals = subclasses_intensity_map[classname]
          else:
            label_intensity_vals = classes_intensity_map[classname]
          masks.append(np.logical_or.reduce([mask == int(liv) for liv in label_intensity_vals]))
        except:
          pass
        
    # Extract each class from label data
    masks = []
    for classname in self.valid_classes:
      try:
        if self.use_subclasses:
          label_intensity_vals = subclasses_intensity_map[classname]
        else:
          label_intensity_vals = classes_intensity_map[classname]
        masks.append(np.logical_or.reduce([mask == int(liv) for liv in label_intensity_vals]))
      except:
        # If this class doesn't exist in the sample, add an empty array of size(mask)
        masks.append(np.zeros_like(mask))
    mask = np.stack(masks, axis=-1).astype('float')
    
    # Extract each class from label data
    masks = []
    for classname in self.valid_classes:
      try:
        if self.use_subclasses:
          label_intensity_vals = subclasses_intensity_map[classname]
        else:
          label_intensity_vals = classes_intensity_map[classname]
        masks.append(np.logical_or.reduce([mask == int(liv) for liv in label_intensity_vals]))
      except:
        # If this class doesn't exist in the sample, add an empty array of size(mask)
        masks.append(np.zeros_like(mask))
    mask = np.stack(masks, axis=-1).astype('float')

    
    # apply augmentations
    if self.augmentation:
      sample = self.augmentation(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    
    # apply preprocessing
    if self.preprocessing:
      sample = self.preprocessing(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    
    return image, mask
    