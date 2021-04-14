"""
python implementation of the semi-global matching algorithm from Stereo Processing by Semi-Global Matching
and Mutual Information (https://core.ac.uk/download/pdf/11134866.pdf) by Heiko Hirschmuller.

author: David-Alexandre Beaupre
date: 2019/07/12
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import pdb

import cv2
import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import zoom
import argparse
import sys
import time as t
from skimage import color,io,img_as_float


from wildcat_model import WILDCAT

PATCH_SIZE = (64, 64)
WEIGHTS_PATH = 'WILDCAT_RES_WEIGHTS.h5'
MODEL = WILDCAT(PATCH_SIZE)
MODEL.load_weights(WEIGHTS_PATH)


def grayscale_and_normalize(im):
  im = color.rgb2gray(im)
  if np.max(im) > 1.0:
    return im/(1.0 * np.iinfo(im.dtype).max)
  else:
    return im
      
def read_images(*file_paths):
  output_list = [grayscale_and_normalize(io.imread(fpath)) for fpath in file_paths]
  return output_list if len(file_paths) > 1 else output_list[0]

class Direction:
  def __init__(self, direction=(0, 0), name='invalid'):
    """
    represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
    :param direction: (x, y) for cardinal direction.
    :param name: common name of said direction.
    """
    self.direction = direction
    self.name = name

# 8 defined directions for sgm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')

class Paths:
  def __init__(self):
    """
    represent the relation between the directions.
    """
    self.paths = [N, NE, E, SE, S, SW, W, NW]
    self.size = len(self.paths)
    self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]

SGMPaths = Paths

class Parameters:
  def __init__(self, min_disparity=0, max_disparity=64, P1=5, P2=70, wsize=(64,64), csize=(7, 7), bsize=(3, 3)):
    """
    represent all parameters used in the sgm algorithm.
    :param max_disparity: maximum distance between the same pixel in both images.
    :param P1: penalty for disparity difference = 1
    :param P2: penalty for disparity difference > 1
    :param csize: size of the kernel for the census transform.
    :param bsize: size of the kernel for blurring the images and median filtering.
    """
    self.max_disparity = max_disparity
    self.min_disparity = min_disparity
    self.P1 = P1
    self.P2 = P2
    self.csize = csize
    self.wsize = wsize
    self.bsize = bsize

SGMParameters = Parameters
CATSGMParameters = SGMParameters

def load_images(left_name, right_name, parameters):
  """
  read and blur stereo image pair.
  :param left_name: name of the left image.
  :param right_name: name of the right image.
  :param parameters: structure containing parameters of the algorithm.
  :return: blurred left and right images.
  """
  left = cv2.imread(left_name, 0)
  left = cv2.GaussianBlur(left, parameters.bsize, 0, 0)
  right = cv2.imread(right_name, 0)
  right = cv2.GaussianBlur(right, parameters.bsize, 0, 0)
  return left, right


def get_indices(offset, dim, direction, height):
  """
  for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
  :param offset: difference with the main diagonal of the cost volume.
  :param dim: number of elements along the path.
  :param direction: current aggregation direction.
  :param height: H of the cost volume.
  :return: arrays for the y (H dimension) and x (W dimension) indices.
  """
  y_indices = []
  x_indices = []

  for i in range(0, dim):
    if direction == SE.direction:
      if offset < 0:
        y_indices.append(-offset + i)
        x_indices.append(0 + i)
      else:
        y_indices.append(0 + i)
        x_indices.append(offset + i)

    if direction == SW.direction:
      if offset < 0:
        y_indices.append(height + offset - i)
        x_indices.append(0 + i)
      else:
        y_indices.append(height - i)
        x_indices.append(offset + i)

  return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset, parameters):
  """
  part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
  given direction)
  :param slice: M x D array from the cost volume.
  :param offset: ignore the pixels on the border.
  :param parameters: structure containing parameters of the algorithm.
  :return: M x D array of the minimum costs for a given slice in a given direction.
  """
  other_dim = slice.shape[0]
  disparity_dim = slice.shape[1]

  disparities = [d for d in range(disparity_dim)] * disparity_dim
  disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

  penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
  penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
  penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

  minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
  minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

  for i in range(offset, other_dim):
    previous_cost = minimum_cost_path[i - 1, :]
    current_cost = slice[i, :]
    costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
    costs = np.amin(costs + penalties, axis=0)
    minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
  return minimum_cost_path


def aggregate_costs(cost_volume, parameters, paths):
  """
  second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
  :param cost_volume: array containing the matching costs.
  :param parameters: structure containing parameters of the algorithm.
  :param paths: structure containing all directions in which to aggregate costs.
  :return: H x W x D x N array of matching cost for all defined directions.
  """
  height = cost_volume.shape[0]
  width = cost_volume.shape[1]
  disparities = cost_volume.shape[2]
  start = -(height - 1)
  end = width - 1

  aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)

  path_id = 0
  for path in paths.effective_paths:
    print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')
    sys.stdout.flush()
    dawn = t.time()

    main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
    opposite_aggregation = np.copy(main_aggregation)

    main = path[0]
    if main.direction == S.direction:
      for x in range(0, width):
        south = cost_volume[0:height, x, :]
        north = np.flip(south, axis=0)
        main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
        opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, parameters), axis=0)

    if main.direction == E.direction:
      for y in range(0, height):
        east = cost_volume[y, 0:width, :]
        west = np.flip(east, axis=0)
        main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
        opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, parameters), axis=0)

    if main.direction == SE.direction:
      for offset in range(start, end):
        south_east = cost_volume.diagonal(offset=offset).T
        north_west = np.flip(south_east, axis=0)
        dim = south_east.shape[0]
        y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
        y_nw_idx = np.flip(y_se_idx, axis=0)
        x_nw_idx = np.flip(x_se_idx, axis=0)
        main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, parameters)
        opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, parameters)

    if main.direction == SW.direction:
      for offset in range(start, end):
        south_west = np.flipud(cost_volume).diagonal(offset=offset).T
        north_east = np.flip(south_west, axis=0)
        dim = south_west.shape[0]
        y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
        y_ne_idx = np.flip(y_sw_idx, axis=0)
        x_ne_idx = np.flip(x_sw_idx, axis=0)
        main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, parameters)
        opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, parameters)

    aggregation_volume[:, :, :, path_id] = main_aggregation
    aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
    path_id = path_id + 2

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

  return aggregation_volume


def compute_costs(left, right, parameters, save_images):
  """
  first step of the sgm algorithm, matching cost based on census transform and hamming distance.
  :param left: left image.
  :param right: right image.
  :param parameters: structure containing parameters of the algorithm.
  :param save_images: whether to save census images or not.
  :return: H x W x D array with the matching costs.
  """
  assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
  assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

  height = left.shape[0]
  width = left.shape[1]
  cheight = parameters.csize[0]
  cwidth = parameters.csize[1]
  y_offset = int(cheight / 2)
  x_offset = int(cwidth / 2)
  disparity = parameters.max_disparity

  left_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
  right_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
  left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
  right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

  print('\tComputing left and right census...', end='')
  sys.stdout.flush()
  dawn = t.time()
  # pixels on the border will have no census values
  for y in range(y_offset, height - y_offset):
    for x in range(x_offset, width - x_offset):
      left_census = np.int64(0)
      center_pixel = left[y, x]
      reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
      image = left[(y - y_offset):(y + y_offset+1), (x - x_offset):(x + x_offset+1)]
      comparison = image - reference
      for j in range(comparison.shape[0]):
        for i in range(comparison.shape[1]):
          if (i, j) != (y_offset, x_offset):
            
            left_census = left_census << 1
            if comparison[j, i] < 0:
              bit = 1
            else:
              bit = 0
            left_census = left_census | bit
      left_img_census[y, x] = np.uint8(left_census)
      left_census_values[y, x] = left_census
      
      right_census = np.int64(0)
      center_pixel = right[y, x]
      reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
      image = right[(y - y_offset):(y + y_offset +1 ), (x - x_offset):(x + x_offset+1)]
      comparison = image - reference
      for j in range(comparison.shape[0]):
        for i in range(comparison.shape[1]):
          if (i, j) != (y_offset, x_offset):
            right_census = right_census << 1
            if comparison[j, i] < 0:
              bit = 1
            else:
              bit = 0
            right_census = right_census | bit
      right_img_census[y, x] = np.uint8(right_census)
      right_census_values[y, x] = right_census

  dusk = t.time()
  print('\t(done in {:.2f}s)'.format(dusk - dawn))

  if save_images:
    cv2.imwrite('left_census.png', left_img_census)
    cv2.imwrite('right_census.png', right_img_census)

  print('\tComputing cost volumes...', end='')
  sys.stdout.flush()
  dawn = t.time()
  left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
  right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
  lcensus = np.zeros(shape=(height, width), dtype=np.int64)
  rcensus = np.zeros(shape=(height, width), dtype=np.int64)
  for d in range(0, disparity):
    rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
    left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
    left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
    while not np.all(left_xor == 0):
      tmp = left_xor - 1
      mask = left_xor != 0
      left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
      left_distance[mask] = left_distance[mask] + 1
    left_cost_volume[:, :, d] = left_distance
    
    lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
    right_xor = np.int64(np.bitwise_xor(np.int64(right_census_values), lcensus))
    right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
    while not np.all(right_xor == 0):
      tmp = right_xor - 1
      mask = right_xor != 0
      right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
      right_distance[mask] = right_distance[mask] + 1
    right_cost_volume[:, :, d] = right_distance

  dusk = t.time()
  print('\t(done in {:.2f}s)'.format(dusk - dawn))

  return left_cost_volume, right_cost_volume


def compute_WILDCAT_costs(left, right, parameters, save_images):
  """
  first step of the sgm algorithm, matching cost based on census transform and hamming distance.
  :param left: left image.
  :param right: right image.
  :param parameters: structure containing parameters of the algorithm.
  :param save_images: whether to save census images or not.
  :return: H x W x D array with the matching costs.
  """
  assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
  assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

  height = left.shape[0]
  width = left.shape[1]
  win_height = parameters.wsize[0]
  win_width = parameters.wsize[1]
  offset = int(win_height / 2)
  disparity = parameters.max_disparity
  
  # Pad images to handle edge cases
  lpadded = np.pad(left, ((offset, offset-1), (offset, offset-1)))
  rpadded = np.pad(right, ((offset, offset-1), (offset, offset-1)))

  # Calculate window views of images
  lwindows = view_as_windows(lpadded, window_shape=(win_height, win_width))
  rwindows = view_as_windows(rpadded, window_shape=(win_height, win_width))
  
  print(' >> Computing cost volume...\n >> ', end='')
  sys.stdout.flush()
  dawn = t.time()
  # Construct cost volume
  cost_volume = np.ones((height, width, disparity))
  for y in range(height):
    for x in range(disparity, width):
      lwin = lwindows[y,x,...]
      batch = np.zeros((disparity, 2, win_height, win_width))
      for d in range(0, disparity):
        batch[d,...] = np.stack((lwin, rwindows[y,np.min(x-d,0), ...]), axis=0)
      similarity_scores = MODEL.predict(batch)
      cost_volume[y,x,:] = -np.squeeze(similarity_scores) # invert similarity scores to make costs
    sys.stdout.flush()
    print('y{}'.format(y),end='.')
  dusk = t.time()
  print(' >> (done in {:.2f}s)'.format(dusk - dawn))
  return cost_volume



def efficient_compute_WILDCAT_costs(left, right, parameters, save_images, ystep=10):

  assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
  assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

  height = left.shape[0]
  width = left.shape[1]
  win_height = parameters.wsize[0]
  win_width = parameters.wsize[1]
  offset = int(win_height / 2)
  max_disp = parameters.max_disparity
  min_disp = parameters.max_disparity
  
  # Pad images to handle edge cases
  lpadded = np.pad(left, ((offset, offset-1), (offset, offset-1)))
  rpadded = np.pad(right, ((offset, offset-1), (offset, offset-1)))
  
  cost = np.zeros((lpadded.shape[1], lpadded.shape[0], max_disp-min_disp))
  for i in range(max_disp-min_disp):
    if i == np.abs(min_disp):
     cost[:, :, i-min_disp] = lpadded
     cost[:, :, i-min_disp] = rpadded
    elif i < 0:
      cost[:, :, i-min_disp] = lpadded
      cost[:, :, i-min_disp] = rpadded
      
  cost = cost.contiguous()
        
  # Calculate window views of images
  #lwindows = view_as_windows(lpadded, window_shape=(win_height, win_width))
  #rwindows = view_as_windows(rpadded, window_shape=(win_height, win_width))
  
  print(' >> Computing cost volume...\n >> ', end='')
  sys.stdout.flush()
  dawn = t.time()
  # Construct cost volume
  cost_volume = np.ones((height, width, disparity))
  for y in range(0, height, ystep):
    for x in range(disparity, width):
      ystp = np.min([ystep, height-y])
      lwin = lwindows[y:y+ystp,x,...]
      batch = np.zeros((disparity*ystp, 2, win_height, win_width))
      for d in range(0, disparity):
        batch[d:d+ystp,...] = np.stack([lwin, rwindows[y:y+ystp,x-d, ...]], axis=1)
      similarity_scores = MODEL.predict(batch)
      cost_volume[y:y+ystp,x,:] = -np.squeeze(similarity_scores.reshape(ystp,disparity)) # negate similarity scores to make costs
    sys.stdout.flush()
    print('y{}'.format(y),end='.')
  dusk = t.time()
  print(' >> (done in {:.2f}s)'.format(dusk - dawn))
  return cost_volume














def very_efficient_compute_WILDCAT_costs(model, left, right, parameters, xstep=4, ystep=4, dstep=4, factor=20):
  assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
  assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'
  assert parameters.max_disparity % dstep == 0, 'disparity must be evenly divisible by dstep'
  assert left.shape[0] % ystep == 0, 'image height must be evenly divisible by ystep'
  assert left.shape[1] % xstep == 0, 'image width must be evenly divisible by xstep'
  
  height = left.shape[0]
  width = left.shape[1]
  win_height = parameters.wsize[0]
  win_width = parameters.wsize[1]
  offset = int(win_height / 2)
  disparity = parameters.max_disparity
  min_disparity = parameters.min_disparity
  
  
  # Pad images to handle edge cases
  lpadded = np.pad(left, ((offset, offset-1), (offset, offset-1)))
  rpadded = np.pad(right, ((offset, offset-1), (offset, offset-1)))

  # Calculate window views of images
  lwindows = view_as_windows(lpadded, window_shape=(win_height, win_width))
  rwindows = view_as_windows(rpadded, window_shape=(win_height, win_width))

  print(' >> Computing cost volume...\n >> ', end='')
  sys.stdout.flush()
  dawn = t.time()

  # Construct cost volume
  cost_volume = np.ones((height//ystep, width//xstep, (disparity-min_disparity)//dstep))
  for y in range(0, height, ystep):
    for x in range(0, width, xstep):
      lwin = lwindows[y,x,...]
      if lwin.max() > 0:
        lwin = lwin/lwin.max()
      
      batch = np.zeros(((disparity-min_disparity)//dstep, 2, win_height, win_width))
      for d in range(0, disparity-min_disparity, dstep):
        rwin = rwindows[y,np.clip(x-d-min_disparity, 0, rwindows.shape[1]-1), ...]
        if rwin.max() > 0:
          rwin = rwin/rwin.max()
          
        batch[d//dstep,...] = np.stack([lwin, rwin], axis=0)
        #print(d, d-min_disparity, d//dstep, disparity, x, y, batch.min(), batch.max())
      similarity_scores = model.predict(batch)
      cost_volume[y//ystep,x//xstep,:] = -np.squeeze(similarity_scores*factor) # negate similarity scores to make costs
      sys.stdout.flush()
    print('y{}'.format(y),end='.')
  dusk = t.time()
  print(' >> (done in {:.2f}s)'.format(dusk - dawn))
  
  #cost_volume = zoom(cost_volume, (ystep, xstep, dstep ))
  return cost_volume
  
WILDCAT_costvolume = very_efficient_compute_WILDCAT_costs














def select_disparity(aggregation_volume):
  """
  last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
  :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
  :return: disparity image.
  """
  volume = np.sum(aggregation_volume, axis=3)
  disparity_map = np.argmin(volume, axis=2)
  return disparity_map


def normalize(volume, parameters):
  """
  transforms values from the range (0, 64) to (0, 255).
  :param volume: n dimension array to normalize.
  :param parameters: structure containing parameters of the algorithm.
  :return: normalized array.
  """
  return 255.0 * volume / parameters.max_disparity


def get_recall(disparity, gt, dims, args):
  """
  computes the recall of the disparity map.
  :param disparity: disparity image.
  :param gt: path to ground-truth image.
  :param args: program arguments.
  :return: rate of correct predictions.
  """
  gt = np.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))[:dims[0],:dims[1]]
  gt = np.int16(gt / 255.0 * float(args.disp))
  disparity = np.int16(np.float32(disparity) / 255.0 * float(args.disp))
  correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
  return float(correct) / gt.size


def get_disparity_image(left_aggregation_volume, parameters):
  return np.uint8(normalize(select_disparity(left_aggregation_volume), parameters))

def save_costvolume_image(img_path, left_cost_volume, parameters):
  left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), parameters))
  cv2.imwrite(img_path, left_disparity_map)
























def sgm():
  """
  main function applying the semi-global matching algorithm.
  :return: void.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--left', default='cones/im2.png', help='name (path) to the left image')
  parser.add_argument('--right', default='cones/im6.png', help='name (path) to the right image')
  parser.add_argument('--left_gt', default='cones/disp2.png', help='name (path) to the left ground-truth image')
  parser.add_argument('--right_gt', default='cones/disp6.png', help='name (path) to the right ground-truth image')
  parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
  parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
  parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
  parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with 3 pixel error')
  args = parser.parse_args()

  left_name = args.left
  right_name = args.right
  left_gt_name = args.left_gt
  right_gt_name = args.right_gt
  output_name = args.output
  disparity = args.disp
  save_images = args.images
  evaluation = args.eval

  
  
  dawn = t.time()

  paths = Paths()

  for USE_WILDCAT in [True, False]:
    if USE_WILDCAT:
      # Prep Model
      MODEL.predict(np.random.rand(1,2,PATCH_SIZE[0],PATCH_SIZE[1]))
      print('\nLoading images...')
      left, right = read_images(left_name, right_name)
      left = left[:372, :448] 
      right = right[:372, :448] 
      dims = [372, 448] 
      parameters = Parameters(max_disparity=disparity, P1=1, P2=2, csize=(7,7), wsize=PATCH_SIZE, bsize=(3, 3))
      prefix = 'WILDCAT_'
    else:
      print('\nLoading images...')
      parameters = Parameters(max_disparity=disparity, P1=10, P2=120, csize=(7,7), wsize=PATCH_SIZE, bsize=(3, 3))
      left, right = load_images(left_name, right_name, parameters)
      left = left[:372, :448] 
      right = right[:372, :448] 
      dims = [372, 448] 
      prefix = 'census_'
      
    print('\nStarting cost computation...')
    if USE_WILDCAT:
      left_cost_volume = very_efficient_compute_WILDCAT_costs(left, right, parameters, xstep=2, ystep=2, dstep=2)
    else:
      left_cost_volume, _ = compute_costs(left, right, parameters, save_images)
    
    if save_images:
      left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), parameters))
      cv2.imwrite(prefix+'disp_map_left_cost_volume.png', left_disparity_map)
      #right_disparity_map = np.uint8(normalize(np.argmin(right_cost_volume, axis=2), parameters))
      #cv2.imwrite('disp_map_right_cost_volume.png', right_disparity_map)
  
    print('\nStarting left aggregation computation...')
    left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)
    #print('\nStarting right aggregation computation...')
    #right_aggregation_volume = aggregate_costs(right_cost_volume, parameters, paths)
  
    print('\nSelecting best disparities...')
    left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), parameters))
    #right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), parameters))
    if save_images:
      cv2.imwrite(prefix+'left_disp_map_no_post_processing.png', left_disparity_map)
      #cv2.imwrite('right_disp_map_no_post_processing.png', right_disparity_map)
  
    print('\nApplying median filter...')
    left_disparity_map = cv2.medianBlur(left_disparity_map, parameters.bsize[0])
    #right_disparity_map = cv2.medianBlur(right_disparity_map, parameters.bsize[0])
    cv2.imwrite(prefix+'left_final.png', left_disparity_map)
    #cv2.imwrite(f'right_{output_name}', right_disparity_map)
  
    if evaluation:
      print('\nEvaluating left disparity map...')
      recall = get_recall(left_disparity_map, left_gt_name, dims, args)
      print('\tRecall = {:.2f}%'.format(recall * 100.0))
      #print('\nEvaluating right disparity map...')
      #recall = get_recall(right_disparity_map, right_gt_name, args)
      #print('\tRecall = {:.2f}%'.format(recall * 100.0))
  
    dusk = t.time()
    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))


if __name__ == '__main__':
  sgm()