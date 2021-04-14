
def very_efficient_compute_WILDCAT_costs(model, left, right, parameters, xstep=4, ystep=4, dstep=4):
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
  
  # Pad images to handle edge cases
  lpadded = np.pad(left, ((offset, offset-1), (offset, offset-1)))
  rpadded = np.pad(right, ((offset, offset-1), (offset, offset-1)))

  # Calculate window views of images
  lwindows = view_as_windows(lpadded, window_shape=(win_height, win_width))
  rwindows = view_as_windows(rpadded, window_shape=(win_height, win_width))
  pdb.set_trace()
  print(' >> Computing cost volume...\n >> ', end='')
  sys.stdout.flush()
  dawn = t.time()
  # Construct cost volume
  cost_volume = np.ones((height//ystep, width//xstep, disparity//dstep))
  for y in range(0, height, ystep):
    for x in range(disparity//dstep, width, xstep):
      lwin = lwindows[y,x,...]
      batch = np.zeros((disparity//dstep, 2, win_height, win_width))
      for d in range(0, disparity, dstep):
        batch[d//dstep,...] = np.stack([lwin, rwindows[y,x-d, ...]], axis=0)
      similarity_scores = model.predict(batch)
      cost_volume[y//ystep,x//xstep,:] = -np.squeeze(similarity_scores**2) # negate similarity scores to make costs
      sys.stdout.flush()
    print('y{}'.format(y),end='.')
  dusk = t.time()
  print(' >> (done in {:.2f}s)'.format(dusk - dawn))
  
  cost_volume = zoom(cost_volume, (ystep, xstep, dstep ))
  return cost_volume
  
WILDCAT_costvolume = very_efficient_compute_WILDCAT_costs