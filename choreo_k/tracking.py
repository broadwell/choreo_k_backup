def output_alphapose_json(poses_series, figure_type='figures'):
  ap_data = []
  image_count = 0
  for frame in poses_series:
    image_count += 1
    image_id = "image" + str(image_count).zfill(5) + '.png'
    #ap_data[image_id] = []
    for p, pose in enumerate(frame[figure_type]):
      score = float(pose.score() * 10)
      rect_pose = frame[figure_type][p]
      keypoints = []
      for kp in pose.data:
        keypoints.extend([float(kp[0]),float(kp[1]),float(kp[2])])
      #ap_data[image_id].append({'score': score, 'keypoints': keypoints})
      ap_data.append({'image_id': image_id, 'score': score, 'keypoints': keypoints})

  with open("alphapose_output.json", 'w') as jfile:
    json.dump(ap_data, jfile)

#!python2 AlphaPose/PoseFlow/tracker-general.py --in_json alphapose_output.json --out_json alphapose_tracked_output.json --imgdir video_frames/BTS_Fire_Full.mp4/

def add_poseflow_figures(input_detections, json_path):
    
  poses_series = copy.deepcopy(input_detections)  
    
  with open(json_path, 'r') as jfile:
    pf_data = json.load(jfile)
  i = 0
  # NOTE: The image IDs must be sorted properly (not done in original PoseFlow)
  tracked_poses = 0
  image_ids = []
  print("SORTING IMAGE IDS AND GETTING TOTAL TRACKED FIGURES")
  for image_id in pf_data:
    image_ids.append(image_id)
    for figure in pf_data[image_id]:
      if 'idx' in figure:
        tracked_poses = max(int(figure['idx'])-1,tracked_poses)
  print("TOTAL TRACKED POSES",tracked_poses+1)
  #for image_id in pf_data:
  for image_id in sorted(image_ids):
    poses_series[i]['aligned_figures'] = []
    aligned_figures = {}
    for figure in pf_data[image_id]:
      if 'idx' not in figure:
        continue
      idx = figure['idx']
      aligned_figures[int(idx)-1] = figure
    for p in range(0,tracked_poses):
      if p in aligned_figures:
        figure = aligned_figures[p]
        score = figure['scores']
        keypoints = figure['keypoints']
        aligned_keypoints = []
        print("Adding keypoints for figure number",p,"ID is",figure['idx'])
        for k in range(0,len(keypoints),3):
          aligned_keypoints.append([keypoints[k], keypoints[k+1], keypoints[k+2]])
      else:
        aligned_keypoints = []
      this_annotation = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON).set(np.asarray(aligned_keypoints), fixed_score=None)
      poses_series[i]['aligned_figures'].append(this_annotation)
      if aligned_keypoints:
        poses_series[i]['aligned_figures'][poses_series[i]['aligned_figures'].index(this_annotation)].text = str(figure['idx'])
        
    # Poses are usually already flipped by this point. If not, the code below would work.
    #if 'flipped_figures' in poses_series[i]:
    #  poses_series[i]['flipped_figures'][f] = flip_detections(poses_series[i]['aligned_figures'])
    i += 1

  print("Processed",i,"frames")
        
  return poses_series
