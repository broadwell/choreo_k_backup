# Descriptors are [[frame_index, pose_index] ...]
# XXX Add an option to average all frames/poses across a range, or an entire video?
def average_poses(pose_series, descriptors, source_figures='zeroified_figures', flip=True):
  all_poses = []
  for descriptor in descriptors:
    all_poses.append(pose_series[descriptor[0]][source_figures][descriptor[1]].data)
  print(len(all_poses))
  poses_array = np.array(all_poses)
  #avg_array = np.zeros(TOTAL_COORDS, 3)
  avg_array = np.sum(poses_array, axis=0)/len(poses_array)
  #print(avg_array)
  this_annotation = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON).set(avg_array, fixed_score=None)
  if flip:
    this_annotation = flip_detections([this_annotation])[0]
  return this_annotation


#from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

def get_feature_vectors(pose_series, figure_type='aligned_figures'):
  features = []
  descriptors = []
  for f, frame_info in enumerate(pose_series):
    for p, pose_info in enumerate(frame_info[figure_type]):
      pose_matrix = get_pose_matrix(frame_info, p)
      if pose_matrix is not None:
        features.append(pose_matrix)
        descriptors.append([f,p])
        
  return([features, descriptors])

# XXX Perhaps set min_samples by some heuristic, e.g., a fraction or multiple of
# frames per second * median number of people in a frame
def cluster_poses(poses_series, figure_type='aligned_figures', min_samples=50):
  print("Getting feature vectors")
  [poses_features, descriptors] = get_feature_vectors(poses_series, figure_type)
  data_array = np.array(poses_features)
  print(data_array.shape)
  print(len(descriptors))

  print("Fitting OPTICS")
  #labels = DBSCAN(eps=100000).fit_predict(features_array)
  labels = OPTICS(min_samples=min_samples, metric='sqeuclidean').fit_predict(data_array)
    
  return [labels, descriptors]


def get_cluster_averages_and_indices(labels, descriptors, pose_series, figure_type='figures', video_file=None, flip_figures=False):

  label_keys = []
  for label in labels:
    if label != -1 and label not in label_keys:
      label_keys.append(label)
  label_keys.sort()

  cluster_indices = {}
  cluster_averages = {}
  cluster_avg_poses = {}

  total_poses = len(label_keys)
    
  for label in label_keys:
    indices = [j for j, x in enumerate(labels) if x == label]
    descs = [descriptors[indices[k]] for k in range(len(indices))]
    print("CLUSTER",label,"|",len(indices),"POSES")
    cluster_indices[label] = indices
    #for l in indices:
    print(descriptors[indices[0]],"CLUSTER",label,'FIRST POSE')
    #plot_poses(new_poses_series[descriptors[l][0]]['aligned_figures'][descriptors[l][1]])
    if video_file is not None:
      fig = excerpt_pose(video_file, pose_series[descriptors[indices[0]][0]], descriptors[indices[0]][1], show=True, source_figure=figure_type, flip_figures=flip_figures)
    avg_pose = average_poses(pose_series, descs)
    cluster_averages[label] = matrixify_pose(avg_pose.data)
    cluster_avg_poses[label] = avg_pose
    print("CLUSTER",label,'AVERAGE POSE')
    plot_poses(avg_pose)

  nrows = math.ceil(total_poses / 5)
  ncols = 5
    
  fig = plt.figure(figsize=(24,10))
  fig.dpi=100    

  for l, label in enumerate(label_keys):    
    ax = fig.add_subplot(nrows,ncols,l+1)
    ax.set_title("POSE " + str(label),loc="right")
    avg_pose = fig2img(plot_poses(cluster_avg_poses[label], show_axis=False, show=False), 4, 5)
    ax.imshow(avg_pose,)
    #ax.set_aspect('auto')
    ax.set_anchor('NE')
    ax.set_axis_off()
  plt.show()

  fig.savefig("Jennie_SOLO_Lisa_Rhee.poses.png", orientation="landscape", format='png', dpi=300)
  #bbox_inches='tight', 
    
  return [cluster_averages, cluster_indices, cluster_avg_poses]

def find_nearest_pose(pose_matrix, cluster_averages):
  best_corr = 0
  best_label = -1
  for label in cluster_averages:
    corr = mantel(pose_matrix, cluster_averages[label])[0]
    if corr > best_corr:
      best_label = label
      best_corr = corr
  return best_label


def compute_pose_distribution(poses_series, labels, descriptors, figure_type='zeroified_figures', cluster_averages=None):

  label_keys = []
  for label in labels:
    if label != -1 and label not in label_keys:
      label_keys.append(label)
  label_keys.sort()
    
  heatmap = np.zeros((len(label_keys)*CELL_HEIGHT, len(poses_series)), dtype=int)

  closest_matches = {}

  for l, label in enumerate(labels):
    f,p = descriptors[l]
    if label >= 0:
      label_index = label_keys.index(label)
      for r in range(CELL_HEIGHT):
        heatmap[(label_index*CELL_HEIGHT)+r,f] += 2
        closest_matches[(f,p)] = label_index
    else:
      if cluster_averages is None:
        continue
      pose_matrix = matrixify_pose(poses_series[f][figure_type][p].data)
      if pose_matrix is not None:
        match_index = find_nearest_pose(pose_matrix,cluster_averages)
        if match_index in label_keys:
          closest_match = label_keys.index(match_index)
          for r in range(CELL_HEIGHT):
            heatmap[(closest_match*CELL_HEIGHT)+r,f] += 1
            closest_matches[(f,p)] = closest_match   
    
  return heatmap, closest_matches

def condense_labels(labels, cluster_map):
  new_labels = labels
    
  for l, label in enumerate(labels):
    if label != -1 and label in cluster_map:
      new_labels[l] = cluster_map[label]

  return new_labels
