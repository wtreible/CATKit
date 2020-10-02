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
  label_classes = [label_name_data[4] for label_name_data in label_names_list]
  label_sub_classes = [label_name_data[5] for label_name_data in label_names_list]
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