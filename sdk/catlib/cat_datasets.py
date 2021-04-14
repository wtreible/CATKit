import pdb
import cv2
import PIL
import os
import numpy as np
import torch
import torchvision
import albumentations as albu
from cat_utils import get_label_name_maps

class SegmentationDataset(torch.utils.data.Dataset):
  'Segmentation Dataset for basic segmentation example'
  def __init__(self, catdb, img_ids, label_ids, valid_classes, use_subclasses=False, preprocessing_fn=None):
    self.catdb = catdb
    self.img_ids = img_ids
    self.label_ids = label_ids
    self.valid_classes = valid_classes
    self.use_subclasses = use_subclasses
    
    self.augmentation = self._default_augmentation_fn()
    self.preprocessing = self._default_preprocessing_fn(preprocessing_fn)
      
  def _default_preprocessing_fn(self, preprocessing_fn):
    if not preprocessing_fn:
      return None
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
    if self.use_subclasses:
      classes_intensity_map = subclasses_intensity_map
    
    # Extract each class based on intensity from label data
    masks = [(mask == 0)] # Add background in first
    for classname in self.valid_classes:
      try:
        label_intensity_vals = classes_intensity_map[classname]
        masks.append(np.logical_or.reduce([mask == int(liv) for liv in label_intensity_vals]))
      except:
        # If this class doesn't exist in the sample, add an empty array of size(mask)
        masks.append(np.zeros_like(mask))
    mask = np.stack(masks, axis=-1).astype('float')
    
    # Augment the (image, label) pair
    #if self.augmentation:
    #  sample = self.augmentation(image=image, mask=mask)
    #  image, mask = sample['image'], sample['mask']
    
    # Preprocess the (image, label) pair
    if self.preprocessing:
      sample = self.preprocessing(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    
    return image, mask
    

class CATSegmentationDataset(torch.utils.data.Dataset):
  'Segmentation Dataset for CAT segmentation example'
  def __init__(self, catdb, img_ids, label_ids, valid_classes, modifier='default', use_subclasses=False, augmentation=True, preprocessing_fn=None):
    self.catdb = catdb
    self.img_ids = img_ids
    self.label_ids = label_ids
    self.valid_classes = valid_classes
    self.modifier = modifier
    self.use_subclasses = use_subclasses
    if augmentation:
      self.augmentation = self._default_augmentation_fn()
    else:
      self.augmentation = None # self._test_augmentation_fn()
      
    self.transform = True
    self.color_transform = self._default_color_transform()
    self.thermal_transform = self._default_thermal_transform()
    self.mask_transform = self._default_mask_transform()
    
  def _default_color_transform(self):
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
    ])
    return transform
    
  def _default_thermal_transform(self):
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
    ])
    return transform
    
  def _default_mask_transform(self):
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
    ])
    return transform
    
  def _default_augmentation_fn(self):
    train_transform = [
      albu.HorizontalFlip(p=0.5),
      albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
      albu.PadIfNeeded(min_height=480, min_width=640, always_apply=True, border_mode=0),
      albu.RandomCrop(height=480, width=640, always_apply=True)
    ]
    # albu.IAAAdditiveGaussianNoise(p=0.2),
    # albu.IAAPerspective(p=0.5),
    # albu.OneOf([
        # albu.CLAHE(p=1),
        # albu.RandomBrightness(p=1),
        # albu.RandomGamma(p=1),
      # ],
      # p=0.9,
    # ),
    # albu.OneOf([
        # albu.IAASharpen(p=1),
        # albu.Blur(blur_limit=3, p=1),
        # albu.MotionBlur(blur_limit=3, p=1),
      # ],
      # p=0.9,
    # ),
    # albu.OneOf([
        # albu.RandomContrast(p=1),
        # albu.HueSaturationValue(p=1),
      # ],
      # p=0.9,
    return albu.Compose(train_transform, additional_targets={'thermal_image': 'image'})
    
  def _test_augmentation_fn(self):
    test_transform = [
      albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
      albu.RandomCrop(height=512, width=512, always_apply=True),
    ]
    return albu.Compose(test_transform, additional_targets={'thermal_image': 'image'})
    
  def __len__(self):
    return len(self.img_ids)

  def __getitem__(self, index):
    
    image_id = self.img_ids[index]
    label_id = self.label_ids[index]
    
    # Get the corresponding items from the CatDB
    image_item = self.catdb[image_id]
    label_item = self.catdb[label_id]
    
    # Read Images
    img_item_path = image_item['path']
    left_color_img_path = os.path.join(img_item_path, 'left_color_' + self.modifier + '.png')
    left_thermal_img_path = os.path.join(img_item_path, 'left_thermal_' + self.modifier + '.png')
    color_image = cv2.imread(left_color_img_path)
    thermal_image = cv2.imread(left_thermal_img_path, 0)
        
    mask = cv2.imread(label_item['path'], 0)

    # Get classes that map to image intensity values
    classes_intensity_map, subclasses_intensity_map = get_label_name_maps(label_item)
    if self.use_subclasses:
      classes_intensity_map = subclasses_intensity_map
    
    # Extract each class based on intensity from label data
    # masks = [(mask == 0)] # Add background in first
    # for classname in self.valid_classes:
      # try:
        # label_intensity_vals = classes_intensity_map[classname]
        # masks.append(np.logical_or.reduce([mask == int(liv) for liv in label_intensity_vals]))
      # except:
        # # If this class doesn't exist in the sample, add an empty array of size(mask)
        # masks.append(np.zeros_like(mask))
    # mask = np.stack(masks, axis=-1).astype('float')
    
    masks = [(mask == 0)] # Add background in first
    for classname in self.valid_classes:
      try:
        label_intensity_vals = classes_intensity_map[classname]
        masks.append(np.logical_or.reduce([mask == int(liv) for liv in label_intensity_vals]))
      except:
        # If this class doesn't exist in the sample, add an empty array of size(mask)
        masks.append(np.zeros_like(mask))
    mask = np.stack(masks, axis=-1)
    mask = np.squeeze(np.argmax(mask, axis=-1))
    
    # Augment the (image, label) pair
    if self.augmentation:
      sample = self.augmentation(image=color_image, thermal_image=thermal_image, mask=mask)
      color_image, thermal_image, mask = sample['image'], sample['thermal_image'], sample['mask']
    
    # Preprocess the (image, label) pair
    color_image = self.color_transform(color_image)
    thermal_image = self.thermal_transform(thermal_image)
    mask = torch.tensor(mask, dtype=torch.int64)
    inputs = torch.cat((color_image, thermal_image), 0)
    labels = mask
    
    return inputs, labels
    