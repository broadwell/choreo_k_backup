import cv2
import os

def get_video_stats(video_file):
  cap = cv2.VideoCapture(video_file)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  video_framerate = cap.get(cv2.CAP_PROP_FPS)
  return [video_framerate, total_frames]

def reset_images_folder(folder_name):
  if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
  for filename in os.listdir(folder_name):
    file_path = os.path.join(folder_name, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
      os.unlink(file_path)

def get_poses_from_video(video_file, start_seconds=0.0, end_seconds=0.0, max_frames=0, seconds_to_skip=0.0, images_too=False, write_images=False):

  cap = cv2.VideoCapture(video_file)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print('total frames in video:',total_frames)

  video_framerate = cap.get(cv2.CAP_PROP_FPS)
  print('video FPS:',video_framerate)
  frame_duration = 1 / float(video_framerate)

  frame_count = 0.0
  frames_processed = 0
  timecode = 0.0
  skip_until = start_seconds

  pose_output = []

  if write_images:
    reset_images_folder('video_images')

  while(cap.isOpened() and (frame_count < total_frames)):
    ret_val, im = cap.read()

    timecode = frame_count * frame_duration
    frame_count += 1

    if (end_seconds and timecode > end_seconds) or (max_frames and frames_processed >= max_frames):
      return pose_output

    if timecode < start_seconds:
      continue

    if (im is None):
      # Might want to retry here
      # print("Missed a frame, continuing...")
      # For now, we'll count a missed frame as a processed frame
      continue
    
    if seconds_to_skip and timecode < skip_until:
      continue
    else:
      skip_until += seconds_to_skip

    im_height, im_width, im_channels = im.shape

    frame_id = int(round(cap.get(1)))

    # Image doesn't necessarily come in as RGB(A)!
    rgbim = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    pil_image = PIL.Image.fromarray(rgbim)

    detections = detect_one_or_more_images([pil_image], processor)
    flipped_detections = flip_detections(detections)
    zeroified_detections = zeroify_detections(detections)
    
    print("Frame",frame_count,"of",total_frames,round(timecode,2),"figures",len(detections))

    this_frame_data = {'frame_id': frame_count, 'time': timecode, 'figures': detections, 'flipped_figures': flipped_detections, 'zeroified_figures': zeroified_detections}
    if images_too:
      this_frame_data['image'] = rgbim
    if write_images:
      pil_image.save(os.path.join('video_images', 'image' + str(int(frames_processed + 1)).zfill(5) + '.png'), 'PNG')

    pose_output.append(this_frame_data)
    frames_processed += 1

  return pose_output
