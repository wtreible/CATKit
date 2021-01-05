import sys, pdb
from os import listdir
from os.path import isdir, isfile, realpath, dirname, splitext, join as joinpath
from collections.abc import MutableMapping

class CatDB(MutableMapping): 
  
  def __init__(self, basepath=None, missing_str='-'):
    self.db = {}
    self.missing_str = missing_str
    self.basepath = basepath if basepath else self._get_basepath()
    try:
      self._populate_db(self.basepath)
    except FileNotFoundError:
      sys.exit("No CATS data found at {}... exiting.".format(self.basepath))

  def _get_basepath(self):
    return joinpath(dirname(dirname(dirname(realpath(__file__)))), 'data')
    
  def _populate_db(self, basepath):
    item_id = 0
    for location in listdir(basepath):
      location_path = joinpath(basepath, location)
      for scene in listdir(location_path):
        scene_path = joinpath(location_path, scene)
        for arrangement in listdir(scene_path):
          arrangement_path = joinpath(scene_path, arrangement)
          if arrangement == 'params':
            # Handle Scene Params
            for modality in listdir(arrangement_path):
              modality_path = joinpath(arrangement_path, modality)
              if isdir(modality_path):
                for item in listdir(modality_path):
                  item_path = joinpath(modality_path, item)
                  item_name = splitext(item)[0]
                  scene_params = {
                    'name' : item_name,
                    'type' : arrangement,
                    'modality' : modality,
                    'scene' : scene,
                    'location' : location,
                    'path' : item_path
                  }
                  self.db[item_id] = scene_params
                  item_id += 1
          else:
            # Handle Arrangements
            for data_type in listdir(arrangement_path):
              data_type_path = joinpath(arrangement_path, data_type)
              if data_type == 'gt':
              
                # Handle Ground Truth Data
                for gt_type in listdir(data_type_path):
                  gt_type_path = joinpath(data_type_path, gt_type)
                  
                  # Ground Truth Point Data
                  if gt_type == 'points':
                    points_path = joinpath(gt_type_path, listdir(gt_type_path)[0])
                    points_item = {
                      'type' : gt_type,
                      'arrangement' : arrangement,
                      'scene' : scene,
                      'location' : location,
                      'path' : points_path
                    }
                    self.db[item_id] = points_item
                    item_id += 1
                    
                  # Ground Truth Segmentation Labels
                  elif gt_type == 'labels':
                    label_image_path = joinpath(gt_type_path, 'label_image.png')
                    label_names_path = joinpath(gt_type_path, 'label_names.csv')
                    bbox_path = joinpath(gt_type_path, 'bbox.csv')
                    thumbnails_path = joinpath(gt_type_path, 'thumbnails')
                    crop_path =  joinpath(gt_type_path, 'crop')
                    labels_item = {
                      'type' : gt_type,
                      'arrangement' : arrangement,
                      'scene' : scene,
                      'location' : location,
                      'path' : label_image_path if isfile(label_image_path) else self.missing_str,
                      'label_names' : label_names_path if isfile(label_names_path) else self.missing_str,
                      'bbox' : bbox_path,
                      'crop' : crop_path,
                      'thumbnails' : thumbnails_path
                    }
                    self.db[item_id] = labels_item
                    item_id += 1
                  
                  # Ground Truth Disparity
                  elif gt_type == 'disparity':
                    for modality in [mx for mx in listdir(gt_type_path) if mx in ['color', 'cross', 'thermal']]:
                        disp_path = joinpath(gt_type_path, modality)
                        disp_item = {
                          'name' : '{}_{}'.format(gt_type, modality),
                          'type' : gt_type,
                          'modality' : modality,
                          'arrangement' : arrangement,
                          'scene' : scene,
                          'location' : location,
                          'path' : joinpath(disp_path, 'gt_disparity.txt'),
                          'interp_path' : joinpath(disp_path, 'gt_disparity_interp.txt'),
                          'disp_range_path' : joinpath(disp_path, 'disp_range.txt')
                        }
                        self.db[item_id] = disp_item
                        item_id += 1
                  
              elif data_type == 'derived':
                # Skip derived data for now...
                pass
              else:
                for modality in listdir(data_type_path):
                  modality_path = joinpath(data_type_path, modality)
                  for item in listdir(modality_path):
                    item_path = joinpath(modality_path, item)
                    item_name = splitext(item)[0]
                    if item_name == 'mask':
                      # Handle Rectification Masks
                      mask_item = {
                        'type' : item_name,
                        'arrangement' : arrangement,
                        'modality' : modality,
                        'scene' : scene,
                        'location' : location,
                        'path' : item_path
                      }
                      self.db[item_id] = mask_item
                      item_id += 1
                    else:
                      # Handle Regular Images
                      item_side, _, item_hazard = item_name.split('_')
                      image_item = {
                        'name' : item_name,
                        'side' : item_side,
                        'type' : data_type,
                        'hazard' : item_hazard,
                        'arrangement' : arrangement,
                        'modality' : modality,
                        'scene' : scene,
                        'location' : location,
                        'path' : item_path
                      }
                      self.db[item_id] = image_item
                      item_id += 1
                      
  def search(self, key, value):
    return [k for k, v in self.db.items() if key in v.keys() and v[key] == value]
  
  def search_subset(self, subset, key, value):
    return [k for k in subset if key in self.db[k].keys() and self.db[k][key] == value]
    
  def multisearch(self, *keyvalues, fn=all):
    assert(len(keyvalues) % 2 == 0)
    keyvalues = list(zip(keyvalues[0::2], keyvalues[1::2]))
    res = []
    for k, v in self.db.items():
      l = []
      for key, value in keyvalues:
        if value[0] == '!':
          l.append(key in v.keys() and v[key] != value[1:])
        else:
          l.append(key in v.keys() and v[key] == value)
      if fn(l):
        res.append(k)
    return res
  
  def __getitem__(self, key):
    if type(key) in [list,tuple]:
      return [self.db[self.__keytransform__(k)] for k in key]
    return self.db[self.__keytransform__(key)]

  def __setitem__(self, key, value):
    self.db[self.__keytransform__(key)] = value

  def __delitem__(self, key):
    del self.db[self.__keytransform__(key)]

  def __iter__(self):
    return iter(self.db)

  def __len__(self):
    return len(self.db)

  def __keytransform__(self, key):
    return key
    
if __name__ == "__main__":
  # Dry Run Testing
  catdb = CatDB()
  for k, v in catdb.items():
    print (str(k) + ' : ' + str(v))