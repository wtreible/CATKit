# https://matplotlib.org/3.1.1/gallery/event_handling/ginput_manual_clabel_sgskip.html#sphx-glr-gallery-event-handling-ginput-manual-clabel-sgskip-py

import time
import numpy as np
import matplotlib.pyplot as plt

# Package imports
from cat_db import CatDB

class ImageViewer():
  def __init__(self, catdb):
    self.catdb = catdb
  
  def view_by_key(self, key):
    image = self.catdb[key]
    plt.title("Key: {}".format(key), fontsize=16)
    plt.plot([1,2], [3,4])
    plt.show()

  
if __name__ == "__main__":
  catdb = CatDB()
  viewer = ImageViewer(catdb)
  viewer.view_by_key(0)