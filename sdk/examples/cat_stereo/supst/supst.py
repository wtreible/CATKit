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