# Note: If an image is supplied, the detections are exepcted to be non-flipped
# Otherwise, they should be flipped.
def plot_poses(detections, image=None, show=True, savepath="", show_axis=True):
  #skeleton_canvas = openpifpaf.show.canvas()

  skeleton_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6, highlight_invisible=True)

  vis_detections = []
  vis_texts = []
  vis_colors = []
    
  if hasattr(detections, 'data'):
    vis_detections = [detections]
    if hasattr(detections, 'text'):
      vis_texts = [detections.text]
      vis_colors = [int(detections.text)]
  else:
    for pose in detections:
      if pose.data.shape[0] == 0:
        continue
      vis_detections.append(pose)
      if hasattr(pose, 'text'):
        vis_texts.append(pose.text)
        vis_colors.append(int(pose.text))

  if not vis_texts or len(vis_texts) != len(vis_detections):
    vis_texts = None
    vis_colors = None

  with openpifpaf.show.canvas(show=show) as ax:
  #ax = openpifpaf.show.canvas(show=show)
    if image is not None:
      ax.imshow(image)
    else:
      ax.set_aspect('equal')
    skeleton_painter.annotations(ax, vis_detections, texts=vis_texts, colors=vis_colors)
    if show:
      openpifpaf.show.canvas()
    if not show_axis:
      ax.set_axis_off()
    fig = ax.get_figure()
    if savepath != "":
      fig.savefig(savepath)
    
  return fig

# For visualizing individual poses with detection overlays
def excerpt_pose(video_file, frame_poses, figure_index=0, show=False, plot_type='pose', source_figure='figures', flip_figures=False, margin=.2, width=None, height=None, show_axis=True):
  
  if source_figure not in frame_poses or len(frame_poses[source_figure]) == 0:
    return None

  figures_frame = copy.deepcopy(frame_poses)
    
  figures_frame['zeroified_figures'] = zeroify_detections(figures_frame[source_figure], width=width, height=height)

  if flip_figures:
    figures_frame['zeroified_figures'] = flip_detections(figures_frame['zeroified_figures'])

  # This is used to cut out the background image, so it must be in the original (non-zeroified) coordinates
  bbox = get_bbox(figures_frame[source_figure][figure_index].data, False, margin=margin, width=width, height=height)
  figures_frame['zeroified_figures'][figure_index].data = shift_figure(figures_frame['zeroified_figures'][figure_index].data, bbox['marg'], bbox['marg'])
    
  cap = cv2.VideoCapture(video_file)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  video_framerate = cap.get(cv2.CAP_PROP_FPS)

  timecode = figures_frame['time']

  frameno = int(round(timecode * video_framerate))
      
  if frameno > total_frames:
    print(frameno,"IS GREATER THAN TOTAL FRAMES IN VIDEO:",total_frames)
    return None

  cap.set(cv2.CAP_PROP_POS_FRAMES, frameno)
  ret_val, im = cap.read()
    
  # Image doesn't necessarily come in as RGB(A)!
  rgbim = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
  pil_image = PIL.Image.fromarray(rgbim)
  
  #crop_img = rgbim[y:y+h, x:x+w]
  #crop_img = rgbim[int(pbox['ymin']):int(pbox['ymax']), int(pbox['xmin']):int(pbox['xmax'])]
  cropped_image = pil_image.crop((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
  if plot_type == 'delaunay':
    fig = plot_delaunay(figures_frame['zeroified_figures'][figure_index], cropped_image, show=show, show_axis=show_axis)
  else:
    fig = plot_poses(figures_frame['zeroified_figures'][figure_index], cropped_image, show=show, show_axis=show_axis)
    
  return fig


# Overlay all detected poses in a single frame on the full image
def overlay_poses(pil_image, figures_frame, show=False, plot_type='pose', source_figure='figures', show_axis=False, savepath=""):
  #print(len(figures_frame[source_figure]))
  
  #print("RUNNING plot_poses")
  if plot_type == 'delaunay':
    fig = plot_delaunay(figures_frame[source_figure], pil_image, show=show, show_axis=show_axis)
  else:
    fig = plot_poses(figures_frame[source_figure], pil_image, show=show, show_axis=show_axis, savepath=savepath)

  #print("RETURNING FROM overlay_poses")
    
  return fig

GC_INTERVAL = 1000

def overlay_video(video_file, pose_data, plot_type='pose', source_figure='figures', show_axis=False, savedir="", start_frame=0):

  #print("OPENING CAPTURE")
  cap = cv2.VideoCapture(video_file)

  #print("GETTING VIDEO INFO")
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  video_framerate = cap.get(cv2.CAP_PROP_FPS)

  for figures_frame in pose_data[start_frame:len(pose_data)]:

    timecode = figures_frame['time']
    frameno = int(round(timecode * video_framerate))
      
    #if frameno > total_frames:
    #  print(frameno,"IS GREATER THAN TOTAL FRAMES IN VIDEO:",total_frames)
       #return None

    #print("GETTING IMAGE FROM CAPTURE")
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameno)
    ret_val, im = cap.read()
    
    # Image doesn't necessarily come in as RGB(A)!
    rgbim = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    pil_image = PIL.Image.fromarray(rgbim)
    
    savepath = os.path.join(savedir, 'image' + str(frameno+1).zfill(5) + '.png')
    #print("GENERATING OVERLAY",savepath)

    fig = overlay_poses(pil_image, figures_frame, source_figure=source_figure, savepath=savepath)
    
    #print("CLEANING UP PLOTS")
    del im, rgbim, pil_image
    
    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close(fig)
    if frameno % GC_INTERVAL == 0:
      gc.collect()
    
    del fig

# Scale keypoint radii by how much they moved in a video

MIN_MOVE = 200
MAX_MOVE = 1200

def draw_figure(point_weights=None, show=True):
  links = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6],
           [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 12],
           [11, 13], [12, 14], [13, 15], [14, 16]]
  coords = [[160, 510],
            [175, 525],
            [145, 525],
            [200, 520],
            [120, 520],
            [215, 440],
            [105, 440],
            [260, 335],
            [60, 335],
            [285, 215],
            [35, 215],
            [200, 280],
            [120, 280],
            [200, 150],
            [120, 150],
            [200, 25],
            [120, 25]]

  xcoords = [c[0] for c in coords]
  ycoords = [c[1] for c in coords]

  if point_weights.any():
    input_weights = np.copy(point_weights)
  else:
    return None

  input_weights *= (200/input_weights.min())

  #if not show:
  #  return [min(input_weights), max(input_weights)]

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  cm = plt.cm.get_cmap('RdYlBu_r')
  ax.set_aspect('equal', adjustable='box')
  #ax.scatter(xcoords, ycoords, input_weights)
  ax.scatter(x=xcoords, y=ycoords, s=input_weights, c=input_weights, vmin=MIN_MOVE, vmax=MAX_MOVE, cmap=cm)

  for link in links:
    ax.plot([coords[link[0]][0], coords[link[1]][0]], [coords[link[0]][1], coords[link[1]][1]], 'k-')

  ax.set_xlim([-35, 340])
  ax.set_ylim([-20, 570])
    
  ax.set_axis_off()
    
  if show:
    plt.show()
  return fig


# Distance matrix-based comparison tests

from skbio import DistanceMatrix
from IPython.display import display

def viz_dist_matrices(p1, p2):
  dmatrix1 = squareform(get_pose_matrix(p1))
  dmatrix2 = squareform(get_pose_matrix(p2))
  
  print(type(dmatrix1))
  #dm1 = DistanceMatrix(dmatrix1)
  #dm1img = dm1.svg
  #display(dm1img)

  plt.xticks(np.arange(17))
  plt.yticks(np.arange(17))

  plt.imshow(dmatrix1, cmap='viridis', origin='upper')
  plot_poses(p1['flipped_figures'][0])

  plt.xticks(np.arange(17))
  plt.yticks(np.arange(17))

  plt.imshow(dmatrix2, cmap='viridis', origin='upper')
  plot_poses(p2['flipped_figures'][0])
  plt.tight_layout()
    
  print("Similarity:",mantel(dmatrix1, dmatrix2)[0])

  diffmatrix = np.absolute(dmatrix1 - dmatrix2)
  movers = diffmatrix.sum(axis=1)
  plt.xticks(np.arange(17))
  plt.yticks(np.arange(17))
  plt.imshow(diffmatrix, cmap='viridis', origin='upper')
  plt.figure()
  plt.xticks(np.arange(17))
  plt.bar(range(17), movers)
  #plt.pcolor(moversarray)
  print(movers.shape)


CELL_HEIGHT=120

def render_pose_distribution(heatmap, poses_series, labels, descriptors, closest_matches=None, show=True, video_file=None, time_index=None, cell_height=CELL_HEIGHT, xlim=None):
    
  # This will overwrite the heatmap; useful for changing appearance
  # of the plot without recomputing everything
  label_keys = []
  for label in labels:
    if label != -1 and label not in label_keys:
      label_keys.append(label)
  label_keys.sort()

  if closest_matches is not None:

    #heatmap = np.zeros(((max(labels)+1)*cell_height, len(poses_series)), dtype=int)
    if xlim is not None:
      map_end = xlim
    else:
      map_end = len(poses_series)
    print("INITIALIZING HEATMAP WITH LENGTH", map_end)
    heatmap = np.zeros((len(label_keys)*cell_height, map_end), dtype=int)
    
    for l, label in enumerate(labels):
      f,p = descriptors[l]
      if f >= map_end:
        continue
      if label >= 0:
        label_index = label_keys.index(label)
        for r in range(cell_height):
          heatmap[(label_index*cell_height)+r,f] += 2
      else:
        if (f,p) in closest_matches:
          closest_match = closest_matches[(f,p)]
          for r in range(cell_height):
            heatmap[(closest_match*cell_height)+r,f] += 1 
    
  fig = plt.figure(figsize=(12,6), constrained_layout=True)
  fig.dpi=100
  ax = plt.gca()
  im = ax.imshow(heatmap, cmap='viridis_r')
  print(cell_height/2, ((len(label_keys)*cell_height)+cell_height/2, cell_height))
  ax.set_yticks(np.arange(cell_height/2, (len(label_keys)*cell_height)+cell_height/2, cell_height))
  ax.set_yticklabels(np.arange(len(label_keys)))
  #ax.set_title("Pose distribution")
    
  if video_file is not None:
    fps, total_frames = get_video_stats(video_file)
    if fps > 0:
      ax.set_xticks(np.arange(0,len(poses_series[:map_end]), fps*15))
      ax.set_xticklabels([int(poses_series[k]['time']) for k in range(0,map_end,int(round(fps*15)))])
      if time_index is not None:
        ax.axvline(x=time_index*fps, ymin=0, ymax=len(label_keys)*cell_height,color='r')
      else:
        # XXX Temporary, for publication image
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  fig.tight_layout()
  if show:
    plt.show()
    fig.savefig("Jennie_SOLO_Lisa_Rhee.heatmap.png", orientation="landscape", pad_inches=0, format='png', bbox_inches='tight', dpi=300)
        
  return heatmap


DPI = 72

FIG_WIDTH = 18
FIG_HEIGHT = 18

# From http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2img(fig2, w=8, h=8, dpi=DPI):
    
  fig2.dpi=dpi
  fig2.set_size_inches(w, h)
  fig2.tight_layout()
  fig2.gca().set_anchor('NE')

  fig2.canvas.draw()

  # Get the RGBA buffer from the figure
  w,h = fig2.canvas.get_width_height()
  buf = np.frombuffer(fig2.canvas.tostring_argb(), dtype=np.uint8)
  buf.shape = (w,h,4)
 
  # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
  buf = np.roll(buf, 3, axis=2)
    
  # put the figure pixmap into a numpy array
  w, h, d = buf.shape
  im = Image.frombytes("RGBA", (w,h), buf)
  return im


def plot_interpose_similarity(pose_series, frame_means, frame_stdevs, video_file, show=False, min_clip=.2):

  fps, total_frames = get_video_stats(video_file)
  print(fps,total_frames)
  window_length = fps

  timecodes = []
  std_uppers = []
  std_lowers = []
  total_frames = min(len(pose_series), len(frame_means))
  for i, frame in enumerate(pose_series[:total_frames]):
    timecodes.append(max(min_clip, frame['time']))
    std_uppers.append(max(min_clip, min(1,frame_means[i] + frame_stdevs[i])))
    std_lowers.append(max(min_clip, max(0,frame_means[i] - frame_stdevs[i])))

  smoothed_means = smooth_series(np.array(frame_means[:total_frames]))
  smoothed_uppers = smooth_series(np.array(std_uppers))
  smoothed_lowers = smooth_series(np.array(std_lowers))

  means_mean = np.nanmean(np.array(frame_means))
  smoothed_means_mean = np.nanmean(np.asarray(smoothed_means))
  means_stdv = np.nanstd(np.array(frame_means))
  smoothed_means_stdv = np.nanstd(np.asarray(smoothed_means))

  THRESH = .9
    
  frame_means_array = np.asarray(frame_means)

  sim_above_thresh = (frame_means_array > THRESH).sum()

  pct_over_thresh = float((np.asarray(frame_means) > THRESH).sum()) / float(len(frame_means))
  smoothed_pct_over_thresh = float((np.asarray(smoothed_means) > THRESH).sum()) / float(len(smoothed_means))

  print(means_mean, means_stdv, smoothed_means_mean, smoothed_means_stdv, round(pct_over_thresh, 2), round(smoothed_pct_over_thresh, 2))

  if show:
    fig = plt.figure(figsize=(12,6), constrained_layout=True)
    fig.dpi=100
    plt.plot(timecodes, smoothed_means, label="mean similarity")
    plt.plot(timecodes, smoothed_uppers, label="upper stdev", linestyle=':')
    plt.plot(timecodes, smoothed_lowers, label="lower stdev", linestyle=':')
    plt.show()
    
    #fig.savefig("BTS_Fire_Full.similarity.png", orientation="landscape", pad_inches=0, format='png', bbox_inches='tight', dpi=300)

  return [smoothed_means, smoothed_uppers, smoothed_lowers, timecodes]



