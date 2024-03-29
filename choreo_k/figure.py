import copy
TOTAL_COORDS = 17
D_THRESH = 0.01

coco_points = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

coco_pts_short = {
    0: 'nose',
    1: 'l_eye',
    2: 'r_eye',
    3: 'l_ear',
    4: 'r_ear',
    5: 'l_shldr',
    6: 'r_shldr',
    7: 'l_elbow',
    8: 'r_elbow',
    9: 'l_wrist',
    10: 'r_wrist',
    11: 'l_hip',
    12: 'r_hip',
    13: 'l_knee',
    14: 'r_knee',
    15: 'l_ankle',
    16: 'r_ankle'
}


def get_figure_coords(coords_and_confidence, margin=0):

  xmin = None
  ymin = None
  xmax = None
  ymax = None
    
  for coord in coords_and_confidence:
    if coord[2] == 0: # and coord[0] == 0 and coord[1] == 0
      continue
    if xmin is None:
      xmin = coord[0]
    if ymin is None:
      ymin = coord[1]
    if xmax is None:
      xmax = coord[0]
    if ymax is None:
      ymax = coord[1]
    xmin = min(xmin, coord[0])
    ymin = min(ymin, coord[1])
    xmax = max(xmax, coord[0])
    ymax = max(ymax, coord[1])
    
  marg = 0
    
  if margin != 0:
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    x_area = area + area * margin
    marg = int(round(math.sqrt(x_area) * margin))
    
    xmax = xmax + marg
    xmin = xmin - marg
    ymax = ymax + marg
    ymin = ymin - marg

  xmed = (xmax + xmin) / 2
  ymed = (ymax + ymin) / 2
    
  return [xmin, ymin, xmax, ymax, xmed, ymed, marg]


def shift_figure(coords_and_confidence, dx, dy):
    
  if coords_and_confidence.shape[0] == 0:
    return coords_and_confidence

  new_cc = np.copy(coords_and_confidence)
  n_rows = coords_and_confidence.shape[0]

  for i in range(0,n_rows):
    xval = coords_and_confidence[i,0]
    yval = coords_and_confidence[i,1]
    conf = coords_and_confidence[i,2]
    if xval == 0 and yval == 0 and conf == 0:
      continue
    new_cc[i,0] = xval + dx
    new_cc[i,1] = yval + dy
    new_cc[i,2] = conf

  return new_cc


# PIL puts the 0,0 origin for images at top left, but the
# distance matrix calculations expect it to be at the bottom left.
# PifPaf wants PIL image input, so the fix for now
# is to flip the coordinates at this point in the pipeline.
# This function flips all detected poses in a single frame.
# If the rectify_x flag is set, this function can also count
# how many coords are on the left or right side of the pose,
# and mirror the coordinates horizontally so that the most
# coords are always on stage right (viewer's left).

def flip_detections(input_detections, flip_y=True, rectify_x=False):

  detections = copy.deepcopy(input_detections)

  for detection in detections:
                
    coords_and_confidence = detection.data
    
    if coords_and_confidence.shape[0] == 0:
      continue

    xmin, ymin, xmax, ymax, xmed, ymed, marg = get_figure_coords(coords_and_confidence)
    
    n_rows = coords_and_confidence.shape[0]

    new_cc = np.copy(coords_and_confidence)

    coords_stage_left = 0
    coords_stage_right = 0
    flip_x=False

    if rectify_x:
      #print("XMED IS",xmed)
      for i in range(0,n_rows):
        xval = coords_and_confidence[i,0]
        yval = coords_and_confidence[i,1]
        conf = coords_and_confidence[i,2]
        if xval == 0 and yval == 0 and conf == 0:
          continue
        if xval > xmed:
          coords_stage_left += 1
        else:
          coords_stage_right += 1
      #print("COORDS STAGE LEFT",coords_stage_left,"COORDS STAGE RIGHT",coords_stage_right)
      if coords_stage_left > coords_stage_right:
        flip_x = True

    for i in range(0,n_rows):
      xval = coords_and_confidence[i,0]
      yval = coords_and_confidence[i,1]
      conf = coords_and_confidence[i,2]
      if xval == 0 and yval == 0 and conf == 0:
        continue
      if flip_y:  
        if yval >= ymed:
          ydiff = yval - ymed
          newy = ymed - ydiff
          new_cc[i,1] = newy
        else:
          ydiff = ymed - yval
          newy = ymed + ydiff
          new_cc[i,1] = newy
      if flip_x:
        if xval > xmed:
          xdiff = xval - xmed
          newx = xmed - xdiff
          new_cc[i,0] = newx
        else:
          xdiff = xmed - xval
          newx = xmed + xdiff
          new_cc[i,0] = newx

    detection.data = new_cc    

  return detections


# Modifies a figure's coordinates so that the corner of its bounding
# box is at 0,0. This is mostly for visualization with PIL images,
# (note that PIL puts y=0 at the top).
# The modifications are done for all figures in a single frame.
def zeroify_detections(input_detections, width=None, height=None):

  detections = copy.deepcopy(input_detections)

  for detection in detections:

    coords_and_confidence = detection.data
    
    if coords_and_confidence.shape[0] == 0:
      continue

    xmin, ymin, xmax, ymax, xmed, ymed, marg = get_figure_coords(coords_and_confidence)

    if width is not None and height is not None:
      dx = (width - (xmax - xmin)) / 2
      dy = (height - (ymax - ymin)) / 2
      
      if dx > 0:
        xmin = xmin - dx
        xmax = xmax + dx
      if dy > 0:
        ymin = ymin - dy
        ymax = ymax + dy
    
    n_rows = coords_and_confidence.shape[0]

    new_cc = np.copy(coords_and_confidence)   

    for i in range(0,n_rows):
      xval = coords_and_confidence[i,0]
      yval = coords_and_confidence[i,1]
      conf = coords_and_confidence[i,2]
      if xval == 0 and yval == 0 and conf == 0:
        continue
      new_cc[i,0] = xval - xmin
      new_cc[i,1] = yval - ymin

    detection.data = new_cc

  return detections


def get_bbox(pose_coords, move_to_origin=False, margin=0, width=None, height=None):
  xmin, ymin, xmax, ymax, xmed, ymed, marg = get_figure_coords(pose_coords, margin)

  if width is not None and height is not None:
    dx = (width - (xmax - xmin)) / 2
    dy = (height - (ymax - ymin)) / 2
      
    if dx > 0:
      xmin = xmin - dx
      xmax = xmax + dx
    if dy > 0:
      ymin = ymin - dy
      ymax = ymax + dy
  #coords_only = pose_coords[:,:2]
  #nonzero_coords = np.ma.masked_equal(coords_only, 0.0, copy=False)
  #xmax, ymax = nonzero_coords.max(axis=0)
  #xmin, ymin = nonzero_coords.min(axis=0)
  if move_to_origin:
    return {'xmax': xmax - xmin, 'xmin': 0, 'ymax': ymax - ymin , 'ymin': 0, 'marg': marg}
  else:
    return {'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin, 'marg': marg}


def get_max_bbox(pose_series, figure_type='zeroified_figures', margin=.1):

  xmax = 0
  ymax = 0
  xmin = 0
  ymin = 0

  for i, frame in enumerate(pose_series):
    for pose in frame[figure_type]:
      bbox = get_bbox(pose.data)
      xmax = max(xmax, bbox['xmax'])
      ymax = max(ymax, bbox['ymax'])
      xmin = min(xmin, bbox['xmin'])
      ymin = min(ymin, bbox['ymin'])
    
  maxwidth = int(math.ceil(xmax - xmin))
  maxheight = int(math.ceil(ymax - ymin))

  maxwidth += int(maxwidth * margin)
  maxheight += int(maxheight * margin)
    
  return [maxwidth, maxheight]


# Get the minimal bounding box of the overlap between two bounding boxes
def get_intersect(a, b):
  minxmax = min(a['xmax'], b['xmax'])
  maxxmin = max(a['xmin'], b['xmin'])
  minymax = min(a['ymax'], b['ymax'])
  maxymin = max(a['ymin'], b['ymin'])
  dx = min(a['xmax'], b['xmax']) - max(a['xmin'], b['xmin'])
  dy = min(a['ymax'], b['ymax']) - max(a['ymin'], b['ymin'])
  if (dx>=0) and (dy>=0):
    return [dx*dy, {'xmin': maxxmin, 'ymin': maxymin, 'xmax': minxmax, 'ymax': minymax}]
  return None


# Get the maximal bounding box around two bounding boxes, if they overlap
def get_union(a, b):
  #print("COMPUTING UNION OF BBOXES A",a,"AND B",b)
  #if a is not None and b is None:
  #  return [get_bbox_area(a), a]
  #elif b is not None and a is None:
  #  return [get_bbox_area[b], b]
  maxxmax = max(a['xmax'], b['xmax'])
  minxmin = min(a['xmin'], b['xmin'])
  maxymax = max(a['ymax'], b['ymax'])
  minymin = min(a['ymin'], b['ymin'])
  dx = min(a['xmax'], b['xmax']) - max(a['xmin'], b['xmin'])
  dy = min(a['ymax'], b['ymax']) - max(a['ymin'], b['ymin'])
  mx = maxxmax - minxmin
  my = maxymax - minymin
  # Return None if there's no overlap
  if (dx>=0) and (dy>=0):
    #print("RETURNING FROM get_union")
    return [mx*my, {'xmin': minxmin, 'ymin': minymin, 'xmax': maxxmax, 'ymax': maxymax}]
  #print("RETURNING NONE FROM get_union")
  return None


def get_bbox_area(bbox):
  width = bbox['xmax'] - bbox['xmin']
  height = bbox['ymax'] - bbox['ymin']
  return width * height


def in_bbox_check(coord, bbox, margin=.5):
  # The margin should be calculated based on the smallest dimension of the pose
  width = bbox['xmax'] - bbox['xmin']
  height = bbox['ymax'] - bbox['ymin']
  mval = min(width,height) * margin

  if coord[0] > (bbox['xmax'] + mval) or coord[0] < (bbox['xmin'] - mval) or coord[1] > (bbox['ymax'] + mval) or coord[1] < (bbox['ymin'] - mval):
    expanded_bbox = [bbox['xmin'] - mval, bbox['ymin'] - mval, bbox['xmax'] + mval, bbox['ymax'] + mval]
    #print("Checked bbox",expanded_bbox)
    return False
  else:
    return True


def average_coords(coord1, coord2):
  return np.array([(coord1[0] + coord2[0])/2, (coord1[1] + coord2[1])/2, (coord1[2] + coord2[2])/2])

def nose_btwn_eyes_ears_shoulders(coords, missing_coords):
  # Nose between eyes
  if 0 in missing_coords and 1 not in missing_coords and 2 not in missing_coords:
    coords[0] = average_coords(coords[1], coords[2])
    missing_coords.remove(0)
  # Nose between ears, interpolate eyes as well
  elif 0 in missing_coords and 3 not in missing_coords and 4 not in missing_coords:
    coords[0] = average_coords(coords[3], coords[4])
    missing_coords.remove(0)
    coords, missing_coords = left_eye_btwn_nose_shoulder(coords, missing_coords)
    coords, missing_coords = right_eye_btwn_nose_shoulder(coords, missing_coords)
  # Nose between shoulders, interpolate eyes and ears as well
  elif 0 in missing_coords and 5 not in missing_coords and 6 not in missing_coords:
    coords[0] = average_coords(coords[5], coords[6])
    missing_coords.remove(0)
    coords, missing_coords = left_eye_btwn_nose_shoulder(coords, missing_coords)
    coords, missing_coords = right_eye_btwn_nose_shoulder(coords, missing_coords)
    coords, missing_coords = left_ear_btwn_eye_shoulder(coords, missing_coords)
    coords, missing_coords = right_ear_btwn_eye_shoulder(coords, missing_coords)
  return [coords, missing_coords]

def left_eye_btwn_nose_shoulder(coords, missing_coords):
  if 1 in missing_coords and 0 not in missing_coords and 5 not in missing_coords:
    coords[1] = average_coords(coords[0], coords[5])
    missing_coords.remove(1)
  return [coords, missing_coords]
        
def right_eye_btwn_nose_shoulder(coords, missing_coords):
  if 2 in missing_coords and 0 not in missing_coords and 6 not in missing_coords:
    coords[2] = average_coords(coords[0], coords[6])
    missing_coords.remove(2)
  return [coords, missing_coords]
    
def left_ear_btwn_eye_shoulder(coords, missing_coords):
  if 3 in missing_coords and 1 not in missing_coords and 5 not in missing_coords:
    coords[3] = average_coords(coords[1], coords[5])
    missing_coords.remove(3)
  return [coords, missing_coords]
    
def right_ear_btwn_eye_shoulder(coords, missing_coords):
  if 4 in missing_coords and 2 not in missing_coords and 6 not in missing_coords:
    coords[4] = average_coords(coords[2], coords[6])
    missing_coords.remove(4)
  return [coords, missing_coords]
    
def left_elbow_btwn_shoulder_wrist(coords, missing_coords):
  if 7 in missing_coords and 5 not in missing_coords and 9 not in missing_coords:
    coords[7] = average_coords(coords[5], coords[9])
    missing_coords.remove(7)
  return [coords, missing_coords]
    
def right_elbow_btwn_shoulder_wrist(coords, missing_coords):
  if 8 in missing_coords and 6 not in missing_coords and 10 not in missing_coords:
    coords[8] = average_coords(coords[6], coords[10])
    missing_coords.remove(8)
  return [coords, missing_coords]
    
def left_hip_btwn_shoulder_knee_ankle(coords, missing_coords):
  if 11 in missing_coords and 5 not in missing_coords and 13 not in missing_coords:
    coords[11] = average_coords(coords[5], coords[13])
    missing_coords.remove(11)
    return [coords, missing_coords]
  elif 11 in missing_coords and 5 not in missing_coords and 15 not in missing_coords:
    coords[11] = average_coords(coords[5], coords[15])
    missing_coords.remove(11) 
  return [coords, missing_coords]

def right_hip_btwn_shoulder_knee_ankle(coords, missing_coords):
  if 12 in missing_coords and 6 not in missing_coords and 14 not in missing_coords:
    coords[12] = average_coords(coords[6], coords[14])
    missing_coords.remove(12)
    return [coords, missing_coords]
  elif 12 in missing_coords and 6 not in missing_coords and 16 not in missing_coords:
    coords[12] = average_coords(coords[6], coords[16])
    missing_coords.remove(12)
  return [coords, missing_coords]
    
def left_ankle_from_knee(coords, missing_coords):
  if 15 in missing_coords and 11 not in missing_coords and 13 not in missing_coords:
    hipdiff = [coords[11][0] - coords[13][0], coords[11][1] - coords[13][1]]
    coords[15] = np.array([coords[13][0] - hipdiff[0], coords[13][1] - hipdiff[1], D_THRESH])
    missing_coords.remove(15)
  return [coords, missing_coords]
    
def right_ankle_from_knee(coords, missing_coords):
  if 16 in missing_coords and 12 not in missing_coords and 14 not in missing_coords:
    hipdiff = [coords[12][0] - coords[14][0], coords[12][1] - coords[14][1]]
    coords[16] = np.array([coords[14][0] - hipdiff[0], coords[14][1] - hipdiff[1], D_THRESH])
    missing_coords.remove(16)
  return [coords, missing_coords]
    
    
# Simple pose correction rules (duh)
def correct_pose(input_coords):
  coords = {}
  weird_coords = []
  for c, coord in enumerate(input_coords):
    #if coord[0] == 0 and coord[1] == 0 and coord[2] <= .01:
    if coord[2] <= D_THRESH:
      weird_coords.append(c)
    coords[c] = np.array([coord[0], coord[1], coord[2]])
  if len(weird_coords) == 0:
    #print("NO COORDS TO CORRECT IN correct_pose")
    return {}
  missing_coords = weird_coords.copy()
  for c in sorted(weird_coords):
    coord = coords[c]
    
    # Find the location of the nose at all costs (between eyes, ears, or shoulders)
    coords, missing_coords = nose_btwn_eyes_ears_shoulders(coords, missing_coords)
    
    # Put the left eye between the nose and the left shoulder (better than the alternative)
    coords, missing_coords = left_eye_btwn_nose_shoulder(coords, missing_coords)
    
    # Put the right eye between the nose and the right shoulder (better than the alternative)
    coords, missing_coords = right_eye_btwn_nose_shoulder(coords, missing_coords)
    
    # Put the left ear between the left eye and the left shoulder (better than the alternative)
    coords, missing_coords = left_ear_btwn_eye_shoulder(coords, missing_coords)
    # Put the right ear between the right eye and the right shoulder (better than the alternative)
    coords, missing_coords = right_ear_btwn_eye_shoulder(coords, missing_coords)
    # Try to put the left shoulder between ear and wrist???
    # Try to put the right shoulder btween ear and wrist???
    
    # Put the left elbow between the left shoulder and the left wrist (better than the alternative)
    coords, missing_coords = left_elbow_btwn_shoulder_wrist(coords, missing_coords)
    # Put the right elbow between the right shoulder and the right wrist (better than the alternative)
    coords, missing_coords = right_elbow_btwn_shoulder_wrist(coords, missing_coords)

    # If wrists are missing, better to let them default to the middle of the pose than extrapolating
    # from shoulder-to-elbow segments (probably?) OR XXX put the wrists on top of the elbows?

    # Put the left hip between the left shoulder and the left knee (better than the alternative)
    # Or between the left shoulder and left ankle
    coords, missing_coords = left_hip_btwn_shoulder_knee_ankle(coords, missing_coords)
    
    # Put the right hip between the right shoulder and the right knee (better than the alternative)
    coords, missing_coords = right_hip_btwn_shoulder_knee_ankle(coords, missing_coords)
    
    # If left ankle is missing, extrapolate from left hip-to-knee segments
    coords, missing_coords = left_ankle_from_knee(coords, missing_coords)
    
    # If right ankle is missing, extrapolate from right hip-to-knee segments
    coords, missing_coords = right_ankle_from_knee(coords, missing_coords)

  corrected_coords = {}
  for w in weird_coords:
    if w not in missing_coords:
      corrected_coords[w] = coords[w]
    
  return corrected_coords

# Determine the maximum number of figures in a single frame of a series,
# the total number of figures (not used), and the last timecode

def count_figures_and_time(input_frames, figure_type='figures'):
  total_figures=0
  max_figures=0
  total_time=0.0
  for i, frame_info in enumerate(input_frames):
    total_figures += len(frame_info[figure_type])
    max_figures = max(max_figures, len(frame_info[figure_type]))
    total_time = max(total_time, frame_info['time'])

  return [max_figures, total_time, total_figures]

# XXX Probably this should vary as a function of the time between frames, rather than the number
# of frames. Possiblly peg it to the number of analyzed frames per second -- assuming that a
# dancer isn't likely to move beyond their previous bounding box within a second.
#FRAME_SEARCH_LIMIT = 5

# Most matrix comparison methods do not allow for empty rows.
# Given a series of poses, fill in each missing point by taking the average of
# its last and next known pcosition (or one of them, if the other is not known).
# Also do this for points with confidence levels < some threshold (50%).
# If no coordinates are available, use the x,y centroid of the poses for that figure (?)
# Note that pose data is grouped by frame and then by figure within each frame.
def interpolate_missing_coords(input_frames, threshold=.5, figure_type='figures', flip_figures=False, check_bbox=False, all_visible=True, overlap_threshold=.7, video_file=None):

  frame_series = copy.deepcopy(input_frames)

  max_figures, total_time, total_figures = count_figures_and_time(frame_series)
  print("MAX FIGURES",max_figures,"TOTAL FIGURES",total_figures)

  frame_search_limit = int(round(float(len(frame_series)) / total_time))
  print(len(frame_series),"FRAMES OVER",total_time,"SECONDS","FPS ROUNDED TO",frame_search_limit)

  for i, frame_info in enumerate(input_frames):
    # Skip the first and last frames? This would change the timing of the analysis -
    # would need to accommodate missing or null frame values
    #if i == 0 or i == len(input_frames)-1:
    #  continue
    if figure_type not in frame_info:
      print("NO FIGURES IN",figure_type,"FOR FRAME",i)
      frame_series[i][figure_type] = []
      continue
    for f in range(len(frame_info[figure_type])):
      if frame_info[figure_type][f].data.shape[0] == 0:
        frame_series[i][figure_type][f].data = np.array([])
        continue
      #print(frame_info[figure_type][f].data)
      #print("EMPTY (NONE) DATA FOR FRAME",i,"FIGURE",f)
      #continue
      new_coords = np.copy(frame_info[figure_type][f].data)
      if new_coords.shape[0] != TOTAL_COORDS:
        print("TRUNCATED FIGURE IN FRAME",i,"FIGURE",f,"NUMBER OF COORDS",new_coords.shape[0])

      bbox = get_bbox(frame_info[figure_type][f].data)
        
      # Maybe put this in a function?
      missing_coords = 0
      confidence_values = []
      for j in range(0,TOTAL_COORDS):
        coord = new_coords[j]
        confidence_values.append(coord[2])
        # Count a coordinate as missing if x,y is 0,0 or if the score is 0 
        if coord[2] == 0:
          missing_coords += 1
      figure_confidence = float(sum(confidence_values)) / float(len(confidence_values))
      if (figure_confidence < .5) or (missing_coords > TOTAL_COORDS / 2):
        print("FRAME",i,"FIGURE",f,"CONFIDENCE",figure_confidence,"MISSING",missing_coords,"COORDS, REMOVING")
        #if f > 0 and f == max_figures-1:
        #print("SUSPECT FIGURE IS LAST IN SERIES, REMOVING FROM RESULTS")
        #del frame_series[i][figure_type][f]
        frame_series[i][figure_type][f].data = np.array([])
        continue

      # XXX Update these as the previous frames are updated? Or just use the original values?
      # If they're updated then there's no need to precompute the paddded bounding boxes above.
      combined_bbox = None
      p_bbox = None
      n_bbox = None
      if i > 0 and figure_type in frame_series[i-1] and frame_series[i-1][figure_type][f].data.shape[0] != 0:
        #p_bbox = frame_figure_bboxes[i-1][f]
        p_bbox = get_bbox(frame_series[i-1][figure_type][f].data, margin=.25)
        combined_bbox = p_bbox
        #print("PREV BBOX PLUS PADDING FOR COMBINED IS",combined_bbox)
      if i < len(frame_series)-1 and figure_type in frame_series[i+1] and frame_series[i+1][figure_type][f].data.shape[0] != 0:
        #n_bbox = frame_figure_bboxes[i+1][f]
        n_bbox = get_bbox(frame_series[i+1][figure_type][f].data, margin=.25)
        combined_bbox = n_bbox
        #print("NEXT BBOX PLUS PADDING FOR COMBINED IS",combined_bbox)

      #combined_bbox = bbox
      if p_bbox is not None and n_bbox is not None:
        combined_data = get_union(p_bbox, n_bbox)
        #print("MERGED PREVIOUS AND NEXT BBOXES, COMBINED IS",combined_bbox)
        if combined_data is None:
          combined_bbox = bbox
          combined_area = 0
          #print("NO PREV OR NEXT BBOX, USING CURRENT FOR COMBINED",combined_bbox)
        else:
          combined_area, combined_bbox = combined_data

      # XXX Do this for the entire pose once, or after each coordinate is checked/upated?
      #corrected_coords = correct_pose(new_coords)
      for j in range(0,TOTAL_COORDS):
        coord = new_coords[j]
        this_coord = None
        # If the score is 0, it may be an occluded point that could be used
        if coord[2] > threshold and combined_bbox is not None and in_bbox_check(coord, combined_bbox):
          #print("USING FRAME",i,"FIGURE",f,"COORD",j,"AS IS")
          continue
        else:
          if all_visible and coord[2] == 0:
            coord = np.array([coord[0], coord[1], D_THRESH])
            #new_coords[j] = np.array([coord[0], coord[1], D_THRESH])
            #continue        
        #print("MISSING/SUSPECT COORD",j,"FOR FRAME",i,"FIGURE",f)
        #print("COORD IS",coord)
        #print("BOUNDING BOX IS",bbox)
        previous_bbox = None
        next_bbox = None
        previous_coord = None
        next_coord = None
        try_extrapolated_position = False
        for p in range(i-1, max(-1,i-frame_search_limit-1), -1):
          # Figures can appear and disappear from frame to frame
          if figure_type in input_frames[p] and f < len(input_frames[p][figure_type]) and input_frames[p][figure_type][f].data.shape[0] != 0 and input_frames[p][figure_type][f].data[j][2] > coord[2]:
            previous_coord = input_frames[p][figure_type][f].data[j]
            #print("FOUND PREVIOUS COORD AT FRAME",p,previous_coord)
            previous_bbox = get_bbox(input_frames[p][figure_type][f].data)
            break
        for n in range(i, min(i+frame_search_limit+1, len(input_frames))):
          # Figures can appear and disappear from frame to frame
          if figure_type in input_frames[n] and f < len(input_frames[n][figure_type]) and input_frames[n][figure_type][f].data.shape[0] != 0 and input_frames[n][figure_type][f].data[j][2] > coord[2]:
            next_coord = input_frames[n][figure_type][f].data[j]
            #print("FOUND NEXT COORD AT FRAME",n,next_coord)
            next_bbox = get_bbox(input_frames[n][figure_type][f].data)
            break
        if previous_coord is not None and next_coord is not None:
          this_coord = average_coords(previous_coord, next_coord)
        elif previous_coord is not None and next_coord is None:
          this_coord = previous_coord
          this_coord[2] += .01
        elif previous_coord is None and next_coord is not None:
          this_coord = next_coord
          this_coord[2] += .01
        
        #print("CHECKING COORD AGAINST EXTENDED BBOX", combined_bbox)
        if this_coord is None or combined_bbox is None or (not in_bbox_check(this_coord, combined_bbox)):
          # Try to extrapolate the missing coord from present coords
          #print("NEXT AND PREVIOUS MATCHES NOT FOUND OR CANDIDATE NOT IN EXTENDED BBOX")
          corrected_coords = correct_pose(new_coords)
          #print('CORRECTED COORDS'," ".join([str(k) for k in corrected_coords.keys()]))
          if j in corrected_coords and (corrected_coords[j][2] > coord[2] or (coord[0] == 0 and coord[1] == 0)):
            #print("TENTATIVELY USING EXTRAPOLATED POSITION")
            this_coord = corrected_coords[j]
            this_coord[2] = max(D_THRESH, this_coord[2])
            try_extrapolated_position = True
          elif coord[2] - D_THRESH <= D_THRESH:
            #print("TENTATIVELY USING FIGURE MIDPOINT")
            this_coord = np.array([(bbox['xmax'] + bbox['xmin'])/2, (bbox['ymax'] + bbox['ymin'])/2, D_THRESH])
            #this_coord = np.array([figure_coords[str(f)]['xmed'], figure_coords[str(f)]['ymed'], .01])
          else:
            #print("USING COORD AS IS FOR NOW")
            continue

        if this_coord is not None:
          #print("CANDIDATE COORD:",this_coord)
          if (not check_bbox) or (previous_bbox is None and next_bbox is None):
            #print("NO NEXT OR PREV BBOX TO CHECK, USING COORD AS IS")
            new_coords[j] = this_coord
          else:
            prev_next_overlap = None
            prev_this_overlap = None
            this_next_overlap = None
            # Check if the previous and next bounding boxes overlap more with each other than
            # either one does with the current frame. If so, the current frame's bounding box
            # is probably an anomaly, and the intersection (or union?) of the previous and next
            # bounding boxes should be used in the check, rather than the current frame's bbox.
            if previous_bbox is not None and next_bbox is not None:
              #print("CHECKING BBOX AGAINST PREVIOUS AND NEXT BBOXES")
              prev_next_overlap = get_intersect(previous_bbox, next_bbox)
              prev_this_overlap = get_intersect(previous_bbox, bbox)
              this_next_overlap = get_intersect(bbox, next_bbox)
              if prev_next_overlap is not None and prev_this_overlap is not None and this_next_overlap is not None:
                if prev_next_overlap[0] > prev_this_overlap[0] and prev_next_overlap[0] > this_next_overlap[0]:
                  this_bbox = prev_next_overlap[1]
                else:
                  this_bbox = bbox
              else:
                this_bbox = bbox
            # If we only have the current and previous or current and next bounding boxes,
            # check that their overlap area is within some threshold of their mean areas
            elif previous_bbox is not None:
              #print("CHECKING BBOX AND PREVIOUS BBOX OVERLAP")
              prev_this_overlap = get_intersect(previous_bbox, bbox)
              this_area = get_bbox_area(bbox)
              prev_area = get_bbox_area(previous_bbox)
              if prev_this_overlap is not None and (prev_this_overlap[0] / ((this_area + prev_area)/2) > overlap_threshold):
                this_bbox = prev_this_overlap[1]
              else:
                this_bbox = bbox
            elif next_bbox is not None:
              #print("CHECKING BBOX AND NEXT BBOX OVERLAP")
              this_next_overlap = get_intersect(bbox, next_bbox)
              this_area = get_bbox_area(bbox)
              next_area = get_bbox_area(next_bbox)
              if this_next_overlap is not None and (this_next_overlap[0] / ((this_area + next_area)/2) > overlap_threshold):
                this_bbox = this_next_overlap[1]
              else:
                this_bbox = bbox
            
            #print("CHECKING COORD AGAINST INTERSECTION BBOX")
            if in_bbox_check(this_coord, this_bbox):
              #print("COORD IN BOUNDING BOX, USING IT")
              new_coords[j] = this_coord
            else:
              #print("NEW COORD NOT IN BOUNDING BOX")
              if try_extrapolated_position and j in corrected_coords and in_bbox_check(corrected_coords[j], this_bbox):
                #print("USING EXTRAPOLATED POSITION")
                this_coord = corrected_coords[j]
              else:
                #print("DEFAULTING TO MIDPOINT")
                this_coord = np.array([(bbox['xmax'] + bbox['xmin'])/2, (bbox['ymax'] + bbox['ymin'])/2, D_THRESH])
                new_coords[j] = this_coord

      frame_series[i][figure_type][f].data = new_coords

    if flip_figures:
      frame_series[i]['flipped_figures'] = flip_detections(frame_series[i][figure_type])
      frame_series[i]['rectified_figures'] = flip_detections(frame_series[i][figure_type], rectify_x=True)
    else:
      frame_series[i]['flipped_figures'] = frame_series[i][figure_type]
      frame_series[i]['rectified_figures'] = flip_detections(frame_series[i][figure_type], flip_y=False, rectify_x=True)

    frame_series[i]['zeroified_figures'] = zeroify_detections(frame_series[i][figure_type])

  return frame_series


