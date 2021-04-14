from __future__ import print_function
from time import gmtime, strftime
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import add as AddTensors
from keras.layers import concatenate as ConcatenateTensors
from keras.layers import Input, Convolution2D, Flatten, Dense, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Lambda, Dropout
K.set_image_dim_ordering('th')


###########################
# Debug Printing Function #
###########################

# Currently unused...
dprint = print
def _get_print_function(verbosity):
  def dprint(*a, **kwargs):
    tstamp_f = "[%Y-%m-%d %H:%M:%S]"
    print(strftime(tstamp_f, gmtime()) + '[Info]>',*a)
    if 'debug' in kwargs.keys():
      print(strftime(tstamp_f, gmtime()) + '[Debug]>', kwargs['debug'])
  if verbosity == 0:
    return lambda *a, **kw: None
  elif verbosity == 1:
    return lambda *a, **kw: print('[Info]>',*a)
  elif verbosity == 2:
    return dprint
  else:
    raise ValueError("Verbose can only be set to 0 (silent), 1 (info), or 2 (debug)")
    
#########################
# Custom Network Layers #
#########################
  
NiN = lambda inp, noutps, name='': Convolution2D(noutps, (1,1), activation='relu', name=name+'NiN')(inp)
SplitChannels = lambda tch_input: [Lambda(lambda x: K.expand_dims(x[:,0,:,:], 1))(tch_input), Lambda(lambda x: K.expand_dims(x[:,1,:,:], 1))(tch_input)]

def identity_block(input_tensor, kernel_size, filters, name, strides=(2, 2)):
  filters1, filters2, filters3 = filters
  bn_axis = 1
  x = Convolution2D(filters1, (1, 1), kernel_regularizer=L2, name=name+'_iden1x1A')(input_tensor)
  x = BatchNormalization(axis=bn_axis)(x)
  x = Activation('relu')(x)
  x = Convolution2D(filters2, kernel_size, padding='same', kernel_regularizer=L2, name=name+'_iden3x3B')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = Activation('relu')(x)
  x = Convolution2D(filters3, (1, 1), kernel_regularizer=L2, name=name+'_iden1x1C')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = AddTensors([x, input_tensor])
  x = Activation('relu')(x)
  return x
    
def conv_block(input_tensor, kernel_size, filters, name, strides=(2, 2)):
  filters1, filters2, filters3 = filters
  bn_axis = 1
  x = Convolution2D(filters1, (1, 1), strides=strides, kernel_regularizer=L2, name=name+'_conv1x1A')(input_tensor)
  x = BatchNormalization(axis=bn_axis)(x)
  x = Activation('relu')(x)
  x = Convolution2D(filters2, kernel_size, padding='same', kernel_regularizer=L2, name=name+'_conv3x3B')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = Activation('relu')(x)
  x = Convolution2D(filters3, (1, 1), kernel_regularizer=L2,  name=name+'_conv1x1C')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  shortcut = Convolution2D(filters3, (1, 1), strides=strides, kernel_regularizer=L2, name=name+'_conv1x1SC')(input_tensor)
  shortcut = BatchNormalization(axis=bn_axis)(shortcut)
  x = AddTensors([x, shortcut])
  x = Activation('relu')(x)
  return x

##########################
# Optimization Functions #
##########################

sgd_opt = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)

################
# Regularizers #
################

L2 = l2(0.01)

##################
# Loss Functions #
##################

def class_loss(y_true,y_pred):
  return K.maximum(K.constant(0, dtype='float32'),K.constant(1, dtype='float32')-y_true*y_pred)
  
##############################
# WILDCAT Network Definition #
##############################
  
def WILDCAT(input_dims, verbose=0):
  
  # == Do some setup == #
  dprint = _get_print_function(verbose)
  
  # == Split 2Ch Input == #
  TwoChInput = Input((2,input_dims[0],input_dims[1]))
  inputL, inputR = SplitChannels(TwoChInput)
  
  # == Left Branch == #
  conv0L = conv_block(inputL, 3, [32, 32, 64], name='conv0L', strides=(1, 1))
  conv1L = identity_block(conv0L, 3, [32, 32, 64], name='conv1L')
  conv2L = identity_block(conv1L, 3, [32, 32, 64], name='conv2L')
  DrL = Dropout(0.2)(conv2L)
  
  conv3L = conv_block(DrL, 3, [64, 64, 128], name='conv3L')
  conv4L = identity_block(conv3L, 3, [64, 64, 128], name='conv4L')
  conv5L = identity_block(conv4L, 3, [64, 64, 128], name='conv5L')
  conv6L = identity_block(conv5L, 3, [64, 64, 128], name='conv6L')
  
  # == Right Branch == #
  conv0R = conv_block(inputR, 3, [32, 32, 64], name='conv0R', strides=(1, 1))
  conv1R = identity_block(conv0R, 3, [32, 32, 64], name='conv1R')
  conv2R = identity_block(conv1R, 3, [32, 32, 64], name='conv2R')
  DrR = Dropout(0.2)(conv2R)
  
  conv3R = conv_block(DrR, 3, [64, 64, 128], name='conv3R')
  conv4R = identity_block(conv3R, 3, [64, 64, 128], name='conv4R')
  conv5R = identity_block(conv4R, 3, [64, 64, 128], name='conv5R')
  conv6R = identity_block(conv5R, 3, [64, 64, 128], name='conv6R')
  
  # == Merged Trunk == #
  merged = ConcatenateTensors([conv6L,conv6R], axis=1)
  nin = NiN(merged,128)
  Dr0M = Dropout(0.2)(nin)
  
  conv7M = conv_block(Dr0M, 3, [128, 128, 256], name='conv7M')
  conv8M = identity_block(conv7M, 3, [128, 128, 256], name='conv8M')
  conv9M = identity_block(conv8M, 3, [128, 128, 256], name='conv9M')
  conv10M = identity_block(conv9M, 3, [128, 128, 256], name='conv10M')
  conv11M = identity_block(conv10M, 3, [128, 128, 256], name='conv11M')
  conv12M = identity_block(conv11M, 3, [128, 128, 256], name='conv12M')
  Dr1M = Dropout(0.2)(conv12M)
  
  conv13M = conv_block(Dr1M, 3, [256, 256, 512], name='conv13M')
  conv14M = identity_block(conv13M, 3, [256, 256, 512], name='conv14M')
  conv15M = identity_block(conv14M, 3, [256, 256, 512], name='conv15M')
  
  poolM = AveragePooling2D((7, 7), name='avg_pool')(conv15M)
  flatM = Flatten()(poolM)
  denseM = Dense(512, activation='relu', kernel_regularizer=L2)(flatM)
  score = Dense(1, kernel_regularizer=L2)(denseM)
  
  # == Compile and Return Model == #
  wildcat = Model(inputs=TwoChInput, outputs=score)
  wildcat.compile(optimizer=sgd_opt, loss=class_loss, metrics=[])
  return wildcat
  