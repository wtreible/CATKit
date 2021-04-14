# Default imports
import os, sys, pdb, time, copy, itertools
from pathlib import Path
from collections import defaultdict
import warnings
warnings.simplefilter("ignore", UserWarning)

# ML and CV library imports
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from skimage.color import label2rgb
from sklearn.metrics import confusion_matrix

# Local imports
from models import UNet, EGUNet, RTFNet, TransBottleneck

# External imports
sys.path.append('/external/RTFNet')
from MF_dataset import MF_dataset
import albumentations as albu
import segmentation_models_pytorch as smp

# Import CATLib packages
sys.path.append('/sdk/catlib')
from cat_db import CatDB
from cat_datasets import CATSegmentationDataset
from cat_utils import read_label_names, get_unique_classes, img_float_to_uint8

# Set important paths and defaults
MODE = 'evaluate' # ['pretrain', 'finetune', 'evaluate']
NETWORK = UNet # [UNet, EGUNet, RTFNet]
MODEL_NAME = 'UNet' # ['UNet', 'EGUNet', 'RTFNet']
DEVICE = 'cuda:2'
BATCH_SIZE = 4

EPOCHS = 100
COLORS_FILE_PATH = 'legend/colors.csv'
LEGEND_FILE_PATH = 'legend/legend.png'
MFDATASET_PATH = '/external/RTFNet/ir_seg_dataset/'
MFDATASET_TRAIN_SPLIT = 'train'
MFDATASET_VAL_SPLIT = 'val'
MFDATASET_TEST_SPLIT = 'val'
SCRATCH_BASEPATH = '/sdk/scratch/cat_segmentation/examples'
os.makedirs(SCRATCH_BASEPATH, mode=0o777, exist_ok=True)

# Initialize CatDB
catdb = CatDB()

# Get Training Image IDs
label_img_ids = catdb.multisearch('location','indoor',
                                  'type', 'labels',
                                  'label_names','!-')                                  
cat_aligned_ids = []
for l_id in label_img_ids:
  tmp_item = catdb[l_id]
  cat_aligned_id = catdb.multisearch('arrangement',tmp_item['arrangement'],
                                     'scene',tmp_item['scene'], 
                                     'type', 'left_ct_aligned')
  cat_aligned_ids.append(cat_aligned_id[0])

# Get Validation Image IDs
val_label_img_ids = catdb.multisearch('location','indoor',
                                  'type', 'labels',
                                  'label_names','!-',
                                  'arrangement','!arrangement5',
                                  'arrangement','!arrangement6',
                                  'arrangement','!arrangement7',
                                  'arrangement','!arrangement8',
                                  'arrangement','!arrangement9',
                                  'arrangement','!arrangement10')
val_cat_aligned_ids = []
for l_id in val_label_img_ids:
  tmp_item = catdb[l_id]
  cat_aligned_id = catdb.multisearch('arrangement',tmp_item['arrangement'],
                                     'scene',tmp_item['scene'], 
                                     'type', 'left_ct_aligned')
  
  val_cat_aligned_ids.append(cat_aligned_id[0])

# Get Test Image IDs
test_label_img_ids = catdb.multisearch('location','indoor',
                                      'type', 'labels',
                                      'label_names','!-')
test_cat_aligned_ids = []
for l_id in test_label_img_ids:
  tmp_item = catdb[l_id]
  cat_aligned_id = catdb.multisearch('arrangement',tmp_item['arrangement'],
                                     'scene',tmp_item['scene'], 
                                     'type', 'left_ct_aligned')
  test_cat_aligned_ids.append(cat_aligned_id[0])

# Determine unique classes in the data
label_items = catdb[label_img_ids]
label_names_list = read_label_names(label_items)
unique_classes, unique_subclasses = get_unique_classes(label_names_list)
print ("Unique Classes in the CATS dataset: {}".format(unique_classes))

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
  cv2.rectangle(legend, (210, (i * TEXT_HEIGHT)), (WIDTH, (i * TEXT_HEIGHT) + TEXT_HEIGHT),
    tuple(color), -1)
cv2.imwrite(LEGEND_FILE_PATH, legend)

def train_model(model, model_path, model_name, dataloaders, optimizer, scheduler, num_epochs=25):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    
    since = time.time()
    total_loss = 0
    epoch_samples = 0
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode
      
      for it, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          
          if model_name == 'EGUNet':
            logits,edge,tedge =  model(inputs)
          else:
            logits = model(inputs)
          
          loss = F.cross_entropy(logits, labels)
          if phase == 'train':
            loss.backward()
            optimizer.step()
            lr_this_epo=0
            for param_group in optimizer.param_groups:
              lr_this_epo = param_group['lr']
              
            print('Train: {}, Epoch {}/{}, Iter {}/{}, LR {:4f}, Loss {:4f}'.format(model_name, epoch, EPOCHS, it+1, len(dataloaders[phase]), lr_this_epo, float(loss)))
            total_loss += float(loss)
            epoch_samples += 1
            
      epoch_loss = total_loss / epoch_samples

      # deep copy the model
      if phase == 'val' and epoch_loss < best_loss:
        print("Saving best model")
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        ckpt_path = os.path.join(os.path.join(SCRATCH_BASEPATH, MODE), '{}_ckpt_valloss_{:4f}.pth'.format(MODEL_NAME, best_loss))
        torch.save(best_model_wts, ckpt_path)

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val loss: {:4f}'.format(best_loss))

  # load best model weights then save model
  model.load_state_dict(best_model_wts)
  torch.save(model.state_dict(),  model_path)
  return model
  
# Train the Model
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

in_ch = 4
mf_out_ch = 9 # 9
cat_out_ch = len(CLASSES)

# Stats:
# UNet No Pretraining (CATS) -- Best val loss: 0.211275
# Unet Pretrained (MFD) - Best val loss: 0.167100
# Unet finetuned (CATS) - Best val loss: 0.201647
# EGUNet Pretrained (MFD) -- Best val loss: 0.148540
# EGUNet finetuned (CATS) -- Best val loss: 0.142341

# Create a Segmentation Datasets
cat_train_dataset = CATSegmentationDataset(catdb, cat_aligned_ids, label_img_ids, 
                                           valid_classes=unique_classes)
cat_val_dataset = CATSegmentationDataset(catdb, val_cat_aligned_ids, val_label_img_ids, 
                                         valid_classes=unique_classes) 

mf_train_dataset = MF_dataset(MFDATASET_PATH, MFDATASET_TRAIN_SPLIT)
mf_val_dataset = MF_dataset(MFDATASET_PATH, MFDATASET_VAL_SPLIT)


# Create Dataloaders
cat_train_loader = torch.utils.data.DataLoader(cat_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
cat_valid_loader = torch.utils.data.DataLoader(cat_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)

mf_train_loader = torch.utils.data.DataLoader(mf_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
mf_valid_loader = torch.utils.data.DataLoader(mf_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)

type_dir = os.path.join(SCRATCH_BASEPATH, MODE)
try:
  os.makedirs(type_dir)
except:
  pass
mf_model_path = os.path.join(SCRATCH_BASEPATH, '{}_mf_pretrained_model.pth'.format(MODEL_NAME))
cat_model_path = os.path.join(type_dir, '{}_cat_finetuned_model.pth'.format(MODEL_NAME))

#########################
# PRETRAINING ON IR SEG #
#########################

if MODE == 'pretrain':
  mf_dataloaders = {
    'train': mf_train_loader,
    'val': mf_valid_loader
  }
  model = NETWORK(in_ch, mf_out_ch).to(device)
  #optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
  optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
  
  #exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.5)

  model = train_model(model, mf_model_path, MODEL_NAME, mf_dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS)
  
####################
# FINETUNE ON CATS #
####################

elif MODE == 'finetune':
  cat_dataloaders = {
    'train': cat_train_loader,
    'val': cat_valid_loader
  }
  print(mf_model_path)

  # Load Model trained w/MF Dataset
  model = NETWORK(in_ch, mf_out_ch)
  model.load_state_dict(torch.load(mf_model_path))
  
  # Replace final layer with new classification layer & move to GPU
  if MODEL_NAME != 'RTFNet':
    model.conv_last = nn.Conv2d(64, cat_out_ch, 1)
  else:
    model.deconv5 = model._make_transpose_layer(TransBottleneck, cat_out_ch, 2, stride=2, inplanes=128)

  model.to(device)
  
  # Setup Opt and LR Scheduler
  optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
  
  # Finetune Model with CATS data
  model = train_model(model, cat_model_path, MODEL_NAME, cat_dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS*2)


####################
# EVALUATE ON CATS #
####################

elif MODE == 'evaluate':

  def plot_confusion_matrix(cm, classes,
                            name='classifier',
                            ignore_background=True,
                            normalize=True,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
                            
    if ignore_background:
      cm = cm[1:,1:]
      classes = classes[1:]
      
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if normalize:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
      
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")
        #txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig('{}_cm.png'.format(name))
    
  def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class
    
  # Create Test Dataset
  cat_test_dataset = CATSegmentationDataset(catdb, test_cat_aligned_ids, test_label_img_ids, 
                                            valid_classes=unique_classes, augmentation=False) 
  
  model_name = MODEL_NAME
  testing_results_file = os.path.join('{}_results.txt'.format(model_name))

  # Load Model
  cat_model_path = os.path.join(SCRATCH_BASEPATH, 'finetune', '{}_cat_finetuned_model.pth'.format(model_name))
  model = NETWORK(in_ch, cat_out_ch)
  model.load_state_dict(torch.load(cat_model_path))
  pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  pytorch_total_params = sum(p.numel() for p in model.parameters())

  print('# Trainable Parameters {}'.format(pytorch_trainable_params))
  print('# Total Parameters {}'.format(pytorch_total_params))
  
  # Init Conf Matrix and Results file
  conf_total = np.zeros((cat_out_ch, cat_out_ch))
  with open(testing_results_file, 'w') as f:
    f.write(",".join(CLASSES) + '\n')
  
  with torch.no_grad():

    print(' >> Evaluating {} ...'.format(model_name))
   
    model.eval()
    model.to(device)
    
    # Iterate over test dataset
    total_time = 0.0
    for i in range(len(cat_test_dataset)):
      dawn = time.time()
      test_image, test_mask = cat_test_dataset[i]
      print("   >> Evaluating image: {}/{}".format(i+1, len(cat_test_dataset)))

      
      # Convert test image to a tensor
      test_tensor = torch.from_numpy(np.expand_dims(test_image,0)).float().to(device)
      if model_name == 'EGUNet':
        output,cedge,tedge =  model(test_tensor)
        cedge = cedge.cpu().numpy().squeeze()
        tedge = tedge.cpu().numpy().squeeze()
      else:
        output = model(test_tensor)
      del test_tensor
      
      dusk = time.time()
      total_time += dusk-dawn
      
      gt_mask = test_mask.cpu().numpy().squeeze()
      label =  gt_mask.flatten()
      predicted_mask = output.argmax(1).cpu().numpy().squeeze()
      prediction = predicted_mask.flatten() 
      conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[l for l in range(cat_out_ch)])
      conf_total += conf
      #plot_confusion_matrix(conf, CLASSES, name='{}'.format(i))
      
      # Prep images for writing
      image = np.transpose(test_image.cpu().numpy(),(1,2,0))
      out_img = image[:,:,:3]*255 #np.zeros(predicted_mask.shape + (3,))
      out_thermal = np.tile(image[:,:,3:]*255., (1,1,3))
      
      gt_img = np.zeros(predicted_mask.shape + (3,))
      for val, clr in enumerate(COLORS):
        if val != 0:
          out_img[predicted_mask == val] = clr # * 0.85 + out_img[intensity_mask == val] * 0.15
          out_thermal[predicted_mask == val] = clr
          gt_img[gt_mask == val] = clr
      
      # Write images
      img_output_dir = os.path.join(SCRATCH_BASEPATH, MODEL_NAME)
      try:
        os.makedirs(img_output_dir)
      except:
        pass
      
      if model_name =='EGUNet':
        edge_output_dir = os.path.join(SCRATCH_BASEPATH, 'EdgeNet')
        try:
          os.makedirs(edge_output_dir)
        except:
          pass
        cv2.imwrite(os.path.join(edge_output_dir, 'rgb_edge_{}.png'.format(i)), (cedge*255.).astype(np.uint8))
        cv2.imwrite(os.path.join(edge_output_dir, 'thermal_edge_{}.png'.format(i)), (tedge*255.).astype(np.uint8))
        
      cv2.imwrite(os.path.join(img_output_dir, 'thermal_image_{}_{}.png'.format(model_name, i)), (image[:,:,3:]*255.).astype(np.uint8))
      cv2.imwrite(os.path.join(img_output_dir, 'rgb_image_{}_{}.png'.format(model_name, i)), (image[:,:,:3]*255.).astype(np.uint8))
      cv2.imwrite(os.path.join(img_output_dir, 'thermal_image_{}_{}.png'.format(model_name, i)), (image[:,:,3:]*255.).astype(np.uint8))
      cv2.imwrite(os.path.join(img_output_dir, 'pred_mask_{}_{}.png'.format(model_name, i)), out_img.astype(np.uint8))
      cv2.imwrite(os.path.join(img_output_dir, 'pred_thermal_mask_{}_{}.png'.format(model_name, i)), out_thermal.astype(np.uint8))
      cv2.imwrite(os.path.join(img_output_dir, 'gt_mask_{}_{}.png'.format(model_name, i)), gt_img.astype(np.uint8))
      
    print('Avg Inference Time (ms): {:0.2f}'.format((total_time / len(cat_test_dataset)) * 1000 ))
    plot_confusion_matrix(conf_total, CLASSES, name=model_name)
    
    precision, recall, IoU = compute_results(conf_total)
    with open(testing_results_file, 'a') as f:
      for i in range(len(precision)):
          f.write('%0.4f | %0.4f, ' % (100*recall[i], 100*IoU[i]))
      f.write('\n')
      f.write('%0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU))))
      
      
''' *ADDITIONAL STUFF
# Save output segmentation image
image = np.transpose(test_image.cpu().numpy(), (1,2,0))
mask = np.transpose(np.squeeze(output.cpu().detach().numpy()), (1,2,0))
pdb.set_trace()
gt_mask = test_mask.cpu().numpy()
intensity_mask = np.argmax(mask, axis=-1)

intensity_mask = np.argmax(mask, axis=-1)
gt_intensity_mask = np.argmax(gt_mask, axis=-1)
out_img = np.zeros(intensity_mask.shape + (3,))
gt_img = np.zeros(intensity_mask.shape + (3,))
for val, clr in enumerate(COLORS):
  if val != 0:
    out_img[intensity_mask == val] = clr # * 0.85 + out_img[intensity_mask == val] * 0.15
    gt_img[gt_intensity_mask == val] = clr
    
cv2.imwrite('output/rgb_image_{}_{}.png'.format(model_name, i), (image[:,:,:3]*255.).astype(np.uint8))
cv2.imwrite('output/thermal_image_{}_{}.png'.format(model_name, i), (image[:,:,3]*255.).astype(np.uint8))
cv2.imwrite('output/mask_{}_{:4f}_{}.png'.format(model_name, iou_score, i), out_img.astype(np.uint8))
cv2.imwrite('output/gt_mask_{}_{}.png'.format(model_name, i), gt_img.astype(np.uint8))
'''