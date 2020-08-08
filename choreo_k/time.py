# Smooth a time series via a sliding window average
# From https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

def smooth_series(x,window_len=11,window='flat'):
  if x.ndim != 1:
    raise(ValueError, "smooth only accepts 1 dimension arrays.")

  if x.size < window_len:
    raise(ValueError, "Input vector needs to be bigger than window size.")

  if window_len<3:
    print("WARNING: window length too small for smoothing, returning input data")
    return x

  if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

  s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
  #print(len(s))
  if window == 'flat': #moving average
    w=np.ones(window_len,'d')
  else:
    w=eval('np.'+window+'(window_len)')

  y=np.convolve(w/w.sum(),s,mode='valid')
  #return y[window_len:-window_len+1]
  #return y[:-window_len+1]
  # This might be cheating -- move the window 1/2 width back to avoid lag
  if window_len % 2 == 0:
    return y[int(window_len/2)-1:-(int(window_len/2))]
  else:
    return y[int(window_len/2):-(int(window_len/2))]


# Generate a full time-series pose similarity heatmap for all available
# poses and frames from the video. This code can use either pose
# characterization approach; in practice, the distance matrix-based analyses
# take longer to calculate but are more accurate.

def corr_time_series_matrix(pose_data, method='distance'):
  pose_correlations = []
  for i, pi in enumerate(pose_data):
    print("Comparing frame",i,"to the rest")
    corr_row = []
    if method == 'distance':
      mi = get_pose_matrix(pi)
    else: # method == 'laplacian'
      mi = get_laplacian_matrix(pi)
    for j, pj in enumerate(pose_data):
      if j < i:
        corr_row.append(pose_correlations[j][i])
      elif j == i:
        corr_row.append(float(1))
      else:
        if mi is None:
          corr_row.append(float(0))
        elif method == 'distance':
          mj = get_pose_matrix(pj)
          if mj is None:
            corr_row.append(float(0))
          else:
            corr_row.append(mantel(mi, mj)[0])
        else: # method == 'laplacian'
          mj = get_laplacian_matrix(pj, figure_index=0, figure_type='flipped_figures')
          if mj is None:
            corr_row.append(float(0))
          else:
            corr_row.append(1 - abs(np.subtract(mi.todense(), mj.todense()).sum()))
    pose_correlations.append(corr_row)

  return pose_correlations

def movements_time_series(pose_data, pose_index=-1, method='distance', figure_type='flipped_figures', video_file=None):

  per_frame_movements = []
  frame_timecodes = []

  pose_indices = []

  max_figures, total_time, total_figures = count_figures_and_time(pose_data, figure_type)

  print("FIGURES PER FRAME IN TIME SERIES:",max_figures)

  # Typically the pose index is only specified if you know there's only one dancer
  # (in which case it's always 0)
  if pose_index != -1:
    max_figures = 1

  for f, frame in enumerate(pose_data):
    frame_movements = []
    frame_timecodes.append(frame['time'])
    if f < len(pose_data)-1:
      for p in range(max_figures):
        this_motion = np.nan
        movers = np.array([])
        # figure p must be available for both f and f+1
        # NOTE check p < len(pose_data[f][figure_type]) in case the data for that frame has been
        # truncated to [] (for example if extraneous data from the end of the video has been removed)
        if p < len(pose_data[f][figure_type]) and p < len(pose_data[f+1][figure_type]) and pose_data[f][figure_type][p].data.shape[0] != 0 and pose_data[f+1][figure_type][p].data.shape[0] != 0:
          p1_conf = sum([c[2] for c in pose_data[f][figure_type][p].data]) / float(len(pose_data[f][figure_type][p].data))
          p2_conf = sum([c[2] for c in pose_data[f+1][figure_type][p].data]) / float(len(pose_data[f+1][figure_type][p].data))
          # XXX NEED AN ELSE HERE?
          # ALSO USE A BETTER CRITERION FOR SKIPPING POSES IF CONFIDENCE IS LOW
          if p1_conf > .5 and p2_conf > .5:
            if method == 'distance':
              plot_type = 'distance'
              dmatrix1 = squareform(get_pose_matrix(pose_data[f], p, figure_type))
              dmatrix2 = squareform(get_pose_matrix(pose_data[f+1], p, figure_type))
              #print("MAX DISTANCE IN FRAME",f,dmatrix1.max())
              diffmatrix = np.absolute(dmatrix1 - dmatrix2)
              movers = diffmatrix.sum(axis=1)
              this_motion = movers.sum(axis=0)
              #all_movements.append(movers)
              #frame_movements.append(movers)
            else:
              plot_type = 'delaunay'
              # Per-keypoint movements are not useful for Laplacian comparisons
              movement = compare_laplacians(pose_data[f], pose_data[f+1], p, figure_type)
              # Can we get meaningful movement values if laplacians are of different sizes?
              if movement is not None:
                movers = np.array([1 - movement])

            # DEBUGGING
            if video_file is not None and not np.isnan(this_motion) and this_motion > 8.5:
              print("FRAME",f,"POSE",p,"CONFIDENCE",p1_conf)
              fig = excerpt_pose(video_file, pose_data[f], p, show=True, source_figure=figure_type, plot_type=plot_type)
              print(pose_data[f][figure_type][p].data)
              print("FRAME",f,"POSE",p,"CONFIDENCE",p2_conf)
              print(pose_data[f+1][figure_type][p].data)
              fig = excerpt_pose(video_file, pose_data[f+1], p, show=True, source_figure=figure_type, plot_type=plot_type)
                
        #all_movements.append(this_motion)
        frame_movements.append(movers)    
        
    per_frame_movements.append(frame_movements)

  #np.array(all_movements),
  return [per_frame_movements, frame_timecodes, max_figures]


# Smooth, summarize, visualize movement data for one or more figures across a time series
def process_movement_series(pose_data, pose_index=-1, figure_type='flipped_figures', video_file=None, method='distance', viz=True):

  print("GETTING MOVEMENT TIME SERIES")
  per_frame_results, frame_timecodes, max_figures = movements_time_series(pose_data, pose_index, method, figure_type, video_file)

  print("CALCULATING CHARACTERISTICS OF TIME SERIES")

  window_length = 5
  if video_file is not None:
    fps, total_frames = get_video_stats(video_file)
    window_length = max(window_length, int(round(fps/2.0)))

  movers = [] # To get the aggregate avg movement of each keypoint (not for Laplacian)
  movement_series = []
  frame_times = []

  smoothed_movement_series = []
    
  for j in range(max_figures):
    movement_series.append([])  
    smoothed_movement_series.append([])

  per_frame_movements = []
    
  for f, frame in enumerate(per_frame_results):
    frame_movements = np.zeros(TOTAL_COORDS)
    frame_times.append(frame_timecodes[f])
    for j in range(max_figures):
      if j >= len(frame) or frame[j].shape[0] == 0:
        #print("POSE",j,"HAS NO MOVEMENT DATA")
        movement_series[j].append(np.nan)
      elif method == 'distance':
        frame_movements = np.add(frame_movements, frame[j])
        movement_series[j].append(sum(frame[j]))
        movers.append(np.array(frame[j]))
      else:
        movement_series[j].append(frame[j])
        movers.append(frame[j])
    per_frame_movements.append(frame_movements)

  figure_time_series = np.array(movers)
        
  if method == 'distance':
    #all_movements = figure_time_series.sum(axis=0)
    movement_means = np.nanmean(figure_time_series, axis=0)
    movement_stdevs = np.nanstd(figure_time_series, axis=0)

  # Window length is half of fps (or ~5, whichever is larger)
  for j in range(max_figures):
    smoothed_movement_series[j] = smooth_series(np.array(movement_series[j]),window_length).tolist()

  if viz:
    print("VISUALIZING TIME SERIES CHARACTERISTICS")
    
    if method == 'distance':
      plt.figure()
      plt.xticks(np.arange(TOTAL_COORDS))
      
      plt.bar(np.arange(TOTAL_COORDS), movement_means)# yerr=movement_stdevs)

      #plt.figure()
      #plt.imshow(distance_correlations, cmap='viridis_r', interpolation='nearest', origin='lower', )
      #plt.show()

    #colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    plt.figure()
    for j in range(len(smoothed_movement_series)):
      #color = colors[j % len(colors)]
      if not np.isnan(np.nanmax(smoothed_movement_series[j])):
        plt.plot(frame_times,smoothed_movement_series[j]) #,color)
    plt.show()

  if method == 'distance':
    return [smoothed_movement_series, frame_times, per_frame_movements, movement_means, movement_stdevs]
  else:
    return [smoothed_movement_series, frame_times]


# Get the mean and standard deviation of inter-pose similarities for each frame

def compare_multiple(pose_data, method='distance', figure_type='aligned_figures'):
  frame_means = []
  frame_stdevs = []
  for f, frame in enumerate(pose_data):
    print("Processing frame",f,"of",len(pose_data))
    frame_similarities = []
    for i, figure_i in enumerate(frame[figure_type]):
      for j, figure_j in enumerate(frame[figure_type]):
        if i < j:
          if method == 'distance':
            mi = get_pose_matrix(frame, i)
            mj = get_pose_matrix(frame, j)
            if mi is None or mj is None:
              similarity = np.nan
            else:
              similarity = mantel(mi, mj)[0]
          else: # method == 'laplacian'
            mi = get_laplacian_matrix(frame, i)
            mj = get_laplacian_matrix(frame, j)
            if mi is None or mj is None:
              similarity = np.nan
            else:
              similarity = 1 - abs(np.subtract(mi.todense(), mj.todense()).sum())
          frame_similarities.append(similarity)
    
    frame_means.append(np.nanmean(frame_similarities))
    frame_stdevs.append(np.nanstd(frame_similarities))

  return [frame_means, frame_stdevs]


# Assumes movement_series is an array of inter-frame movement values, one for each
# detected pose, with missing poses identified via np.nan, as generated from
# process_movement_series. 
def average_frame_movements(movement_series, poses_series, show=False, max_clip=3, video_file=None):
  if len(movement_series) == 0:
    print("ERROR: empty movement series")
  # Each row should have the same length, so use the first one
  total_frames = min(len(movement_series[0]), len(poses_series))
  total_poses = len(movement_series)
  frame_means = []
  upper_stdvs = []
  lower_stdvs = []
  timecodes = []
  for frame in poses_series[:total_frames]:
    timecodes.append(frame['time'])
  for f in range(total_frames):
    frame = []
    for p in range(total_poses):
      frame.append(movement_series[p][f])
    if len(frame) == 0 or np.isnan(np.nanmean(frame)):
      frame_stdv = 0
      upper_stdv = 0
      lower_stdv = 0
      frame_mean = 0
    else:
      frame_mean = min(max_clip, np.nanmean(frame))
      frame_stdv = np.nanstd(frame)
      upper_stdv = min(frame_mean + frame_stdv, max_clip)
      lower_stdv = min(frame_mean - frame_stdv, max_clip)
    frame_means.append(frame_mean)
    upper_stdvs.append(upper_stdv)
    lower_stdvs.append(lower_stdv)

  if video_file is not None:
    
    fps, total_frames = get_video_stats(video_file)
    
    mean_mvt = np.nanmean(np.array(frame_means))
    mean_stdv = np.nanstd(np.array(frame_means))
    
    mean_mvt_ps = mean_mvt * fps

    # These are already divided by the number of dancers
    print(mean_mvt, mean_stdv, mean_mvt_ps)

  if show:
    fig = plt.figure(figsize=(12,6), constrained_layout=True)
    fig.dpi=100
    plt.plot(timecodes, frame_means, label="mean movement")
    plt.plot(timecodes, upper_stdvs, label="upper stdev", linestyle=':')
    plt.plot(timecodes, lower_stdvs, label="lower stdev", linestyle=':')
    plt.show()

  #fig.savefig("BTS_Fire_Full.movements.png", orientation="landscape", pad_inches=0, format='png', bbox_inches='tight', dpi=300)
    
  return [frame_means, upper_stdvs, lower_stdvs, timecodes]


def member_frame_movements(movement_series, poses_series, max_clip=3, show=False, condense=True):
  print(len(movement_series),"DANCERS TO CHECK")
  total_frames = min(len(movement_series[0]), len(poses_series))
  # Remove series for dancers who never move (due to clipping of sequence)
  valid_series = []
  for d, dancer in enumerate(movement_series):
    print("LOOKING AT DANCER",d)
    if np.isnan(np.nanmax(dancer[:total_frames])):
      print("DANCER NEVER MOVES, SKIPPING")
      #print(dancer)
      continue
    else:
      for v, val in enumerate(dancer[:total_frames]):
        if val > max_clip:
          dancer[v] = max_clip
      valid_series.append(dancer[:total_frames])
  timecodes = []
  for frame in poses_series[:total_frames]:
    timecodes.append(frame['time'])
  valid_array = np.transpose(np.array(valid_series))
    
  max_frame_figures = 0
  for frame in valid_array:
    max_frame_figures = max(max_frame_figures,np.count_nonzero(~np.isnan(frame)))

  condensed_array = np.zeros((valid_array.shape[0],max_frame_figures),dtype=float)

  figures_per_frame = []
  for f, frame in enumerate(valid_array):
    #print("LENGTH OF FRAME:",len(frame))
    figures_this_frame = np.count_nonzero(~np.isnan(frame))
    max_frame_figures = max(max_frame_figures,figures_this_frame)
    figures_per_frame.append(figures_this_frame)

    non_nans = np.argwhere(~np.isnan(frame))
    for m in range(0,max_frame_figures):
      if m < len(non_nans):
        condensed_array[f,m] = frame[non_nans[m]]
      else:
        condensed_array[f,m] = np.nan
    
  print(max(figures_per_frame))
    
  if show:
    
    fig = plt.figure(figsize=(12,6), constrained_layout=True)
    fig.dpi=100
    if not condense:
      plt.plot(timecodes, valid_array)
    else:
      plt.plot(timecodes, condensed_array)
    plt.show()

  if not condense:
    return valid_array
  else:
    return condensed_array


def nan_helper(y):
  return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nans(series):
  nans, x= nan_helper(series)
  series[nans]= np.interp(x(nans), x(~nans), series[~nans])
  return series
