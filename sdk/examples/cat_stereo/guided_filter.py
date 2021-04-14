import numpy as np
import cv2

class GrayGuidedFilter():

  def __init__(self, I, radius, eps):
      """
      Parameters
      ----------
      I: NDArray
          2D guided image
      radius: int
          Radius of filter
      eps: float
          Value controlling sharpness
      """
      self.I = img
      self.radius = (radius, radius)
      self.eps = eps

  def filter(self, p):
      """
      Parameters
      ----------
      p: NDArray
          Filtering input of 2D
      Returns
      -------
      q: NDArray
          Filtering output of 2D
      """
      # step 1
      meanI  = cv2.blur(self.I, self.radius)
      meanp  = cv2.blur(p, self.radius)
      corrI  = cv2.blur(self.I * self.I, self.radius)
      corrIp = cv2.blur(self.I * p, self.radius)
      # step 2
      varI   = corrI - meanI * meanI
      covIp  = corrIp - meanI * meanp
      # step 3
      a      = covIp / (varI + self.eps)
      b      = meanp - a * meanI
      # step 4
      meana  = cv2.blur(a, self.radius)
      meanb  = cv2.blur(b, self.radius)
      # step 5
      q = meana * self.I + meanb

      return q