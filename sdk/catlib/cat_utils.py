import csv
import numpy as np
  
def read_label_names(label_items, keyname='label_names'):
  '''
  Reads label names CSVs for semantic segmentation labels
  '''
  def _read_one_csv(path):
    with open(path) as fp:
      reader = csv.reader(fp, delimiter=",", quotechar='"')
      data = [row for row in reader]
    return data
  if type(label_items) in [list,tuple]:
    data_concat = []
    for item in label_items:
      data_concat += _read_one_csv(item[keyname])
    return data_concat
  else:
    return _read_one_csv(label_items[keyname])
    
def get_unique_classes(label_names_list, invalid_items=['-']):
  '''
  Returns unique valid class and subclass names from output of "read_label_names"
  '''
  label_classes = [label_name_data[4].strip() for label_name_data in label_names_list]
  label_sub_classes = [label_name_data[5].strip() for label_name_data in label_names_list]
  unique_classes = np.unique(label_classes)
  unique_subclasses = np.unique(label_sub_classes)
  return  np.setdiff1d(unique_classes,invalid_items), np.setdiff1d(unique_subclasses,invalid_items)

def get_label_name_maps(label_item, invalid_items=['-'], keyname='label_names'):
  '''
  Reads label names CSVs into a map of class_name : label_image intensity value
    from output of "read_label_names"
  '''
  def _try_append(class_map, key, value):
    try:
      class_map[key].append(value)
    except KeyError:
      class_map[key] = [value]
    return class_map
  label_names = read_label_names(label_item, keyname=keyname)
  classes_map = {}
  subclasses_map = {}
  for label_name_data in label_names:
    if label_name_data[4] not in invalid_items:
      classes_map = _try_append(classes_map, label_name_data[4], int(label_name_data[3]))
    if label_name_data[5] not in invalid_items:
      subclasses_map = _try_append(subclasses_map, label_name_data[5], int(label_name_data[3]))
  return classes_map, subclasses_map
  
def img_float_to_uint8(image):
  '''
  Converts a float image to a uint8 image for saving
  '''
  return ((image-image.min())/(image.max()-image.min()) * 255.).astype(np.uint8)
  
def pprint(d, indent=0, indent_size=2):
  '''
  Pretty prints a dictionary
  '''
  indent_str = ' ' * indent_size
  for key, value in d.items():
    print(indent_str * indent + str(key))
    if isinstance(value, dict):
      pprint(value, indent=indent+1, indent_size=indent_size)
    else:
      print(indent_str * (indent+1) + str(value))

def mdprint(*items, keys=[], sep=':', end=' '):
  '''
  Multi-descriptor print: 
    > Prints multiple values from a dictionary for the given set of keys 
    > Values are separated by 'sep'
    > Items are separated by 'end'
  '''
  tmp_str = ''
  for item in items:
    for key in keys:
      tmp_str += item[key] + sep
    tmp_str = tmp_str[:-len(sep)] + end
  print(tmp_str)