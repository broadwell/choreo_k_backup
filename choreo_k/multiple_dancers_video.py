from scipy import stats

DPI = 72

FIG_WIDTH = 16
FIG_HEIGHT = 12

XLIM = 10550

CELL_HEIGHT=230

VIDEO_FILE = 'BTS_Fire_Full.mp4'

fps, frames = get_video_stats(VIDEO_FILE)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
REALLY_BIG_SIZE = 28

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

out_path = '/srv/choreo/bts_video/'
#reset_images_folder(out_path)

plt.ioff()

START_FRAME = 0

margin = .1

# Probably don't need to do this, since we're not plotting individual excerpts

label_keys = []
for label in labels:
  if label != -1 and label not in label_keys:
    label_keys.append(label)
label_keys.sort()

# XXX This should have been done in compute_pose_distribution ...
label_matches = {}

for l, label in enumerate(labels):
  f,p = descriptors[l]
  if label >= 0:
    label_index = label_keys.index(label)
    label_matches[(f,p)] = label_index
    
#poses_heatmap = heatmap
heatmap = render_pose_distribution(heatmap, new_poses_series[0:XLIM], labels, descriptors, closest_matches=closest_matches, video_file=VIDEO_FILE, show=False, cell_height=CELL_HEIGHT)

interpose_means, interpose_uppers, interpose_lowers, timecodes = plot_interpose_similarity(new_poses_series[:XLIM], frame_means[:XLIM], frame_stdevs[:XLIM], VIDEO_FILE, show=False)
    
frame_mvt_means, frame_mvt_uppers, frame_mvt_lowers, timecodes = average_frame_movements(smoothed_movement_series, new_poses_series[:XLIM], show=False)

member_series = member_frame_movements(smoothed_movement_series, new_poses_series[:XLIM])

fig = plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))
fig.dpi=DPI
fig.tight_layout()

for f, frame in enumerate(new_poses_series[0:XLIM]):
  
  if f < START_FRAME:
    continue
    
  print('PROCESSING FRAME',f,"OF",len(new_poses_series[:XLIM])-1,"FRAMES, TIME",frame['time'])

  video_frame_path = os.path.join('/srv/choreo/bts_frames', 'image' + str(f+1).zfill(5) + '.png')
  if not os.path.isfile(video_frame_path):
    print("ERROR: missing frame file for",f+1)
    break

  print("GETTING OVERLAY IMAGE")
  frame_img = Image.open(video_frame_path)
    
  avg_matches = []
  poses_found = 0

  print("DETERMINING AVERAGE POSE")
  for p, pose in enumerate(frame['rectified_figures']):
    if pose.data.shape[0] == 0:
      continue
    poses_found += 1
    if (f,p) in label_matches:
      avg_matches.append(label_matches[(f,p)])
    elif (f,p) in closest_matches:
      avg_matches.append(closest_matches[(f,p)])
  cluster_pose_index = stats.mode(avg_matches).mode[0]

  cluster_pose_label = label_keys[cluster_pose_index]

  # NOTE: cluster_avg_poses wants the real label number (after condensing), not the index of this number
  # in label_keys (i.e., its row in the heatmap), which is what label_matches and closest_matches return
  avg_pose = fig2img(plot_poses(cluster_avg_poses[cluster_pose_label], show_axis=False, show=False), 4, 6)

  print("DRAWING STICK FIGURE")
  stick_figure = fig2img(draw_figure(per_frame_movements[f], show=False), 4, 6)

  if f == START_FRAME:

      print("PLOTTING OVERLAY IMAGE")
      ax1 = fig.add_subplot(3,2,1)
      ax1.set_title("VIDEO FRAME " + str(f))
      ax1.set_anchor('NW')
      overlay_image = ax1.imshow(frame_img)
      ax1.set_axis_off()

      print("PLOTTING TEXT")
      ax7 = fig.add_subplot(3,6,4)
      ax7.set_title("TIME: " + str(round(frame['time'],2)) + "s", loc='left')
      info_text = "\nDETECTED\nPOSES: " + str(poses_found) + "\n\nMOST FREQUENT\nKEY POSE: " + str(cluster_pose_index) + "\n"
      info_text += "\nINTER-FRAME\nMOVEMENT: " + str(round(frame_mvt_means[f],2)) + "\n\n" + "INTER-POSE\nSIMILARITY: " + str(round(interpose_means[f],2))
      ax7.text(0., 1, info_text, verticalalignment='top', horizontalalignment='left')
      ax7.set_axis_off()

      print("PLOTTING HEATMAP")
      ax4 = fig.add_subplot(3,2,3)
      ax4.set_title("KEY POSES DISTRIBUTION")
      ax4.set_anchor('W')
      ax4.plot(heatmap)
      im = ax4.imshow(heatmap, cmap='viridis_r')
      ax4.set_yticks(np.arange(CELL_HEIGHT/2, (len(label_keys)*CELL_HEIGHT)+CELL_HEIGHT/2, CELL_HEIGHT))
      ax4.set_yticklabels(np.arange(len(label_keys)))
      ax4.set_xticks(np.arange(0,len(new_poses_series[:XLIM]), fps*20))
      ax4.set_xticklabels([int(new_poses_series[k]['time']) for k in range(0,len(new_poses_series[:XLIM]),int(math.ceil(fps*20)))])
      heatmap_time = ax4.axvline(x=frame['time']*fps, ymin=0, ymax=len(label_keys)*CELL_HEIGHT,color='r')

      print("PLOTTING KEY POSE")
      ax2 = fig.add_subplot(3,6,5)
      ax2.set_anchor('SW')
      keypose_image = ax2.imshow(avg_pose)
      ax2.set_title("KEY POSE " + str(cluster_pose_index), loc="left")
      ax2.set_axis_off()

      print("PLOTTING STICK FIGURE")
      ax3 = fig.add_subplot(3,6,6)
      stick_figure_image = ax3.imshow(stick_figure)
      ax3.set_title("KEYPOINT MOVEMENT")
      ax3.set_axis_off()

      print("PLOTTING INTER-POSE MOVEMENT")
      ax5 = fig.add_subplot(3,2,4)
      ax5.set_title("INTER-FRAME MOVEMENT")
      ax5.plot(timecodes, frame_mvt_means, label="mean movement")
      ax5.plot(timecodes, frame_mvt_uppers, label="upper stdev", linestyle=':')
      ax5.plot(timecodes, frame_mvt_lowers, label="lower stdev", linestyle=':')
      ax5.margins(x=0)
      mvt_time = ax5.axvline(x=frame['time'], ymin=0, ymax=4,color='r')
    
      print("PLOTTING INDIVIDUAL MOVEMENT TRACKS")
      ax8 = fig.add_subplot(3,2,5)
      ax8.set_title("INDIVIDUAL POSE MOVEMENT")
      ax8.plot(timecodes, member_series)
      ax8.margins(x=0)
      indiv_time = ax8.axvline(x=frame['time'], ymin=0, ymax=3,color='r')

      print("PLOTTING INTER-POSE SIMILARITY")
      ax6 = fig.add_subplot(3,2,6)
      ax6.set_title("INTER-POSE SIMILARITY BY FRAME")
      ax6.plot(timecodes, interpose_means, label="mean similarity")
      ax6.plot(timecodes, interpose_uppers, label="upper stdev", linestyle=':')
      ax6.plot(timecodes, interpose_lowers, label="lower stdev", linestyle=':')
      ax6.margins(x=0)
      interpose_time = ax6.axvline(x=frame['time'], ymin=0, ymax=1,color='r')

  else:

      print("PLOTTING OVERLAY IMAGE")
      ax1.set_title("VIDEO FRAME " + str(f))
      overlay_image.remove()
      overlay_image = ax1.imshow(frame_img)

      ax7.remove()
      ax7 = fig.add_subplot(3,6,4)
      ax7.set_title("TIME: " + str(round(frame['time'],2)) + "s", loc='left')
      info_text = "\nDETECTED\nPOSES: " + str(poses_found) + "\n\nMOST FREQUENT\nKEY POSE: " + str(cluster_pose_index) + "\n"
      info_text += "\nINTER-FRAME\nMOVEMENT: " + str(round(frame_mvt_means[f],2)) + "\n\n" + "INTER-POSE\nSIMILARITY: " + str(round(interpose_means[f],2))
      #ax7.set_anchor('NW')
      ax7.text(0, 1, info_text, verticalalignment='top', horizontalalignment='left')
      ax7.set_axis_off()
    
      heatmap_time.remove()
      heatmap_time = ax4.axvline(x=frame['time']*fps, ymin=0, ymax=len(label_keys)*CELL_HEIGHT,color='r')
    
      ax2.set_title("KEY POSE " + str(cluster_pose_index),loc='left')
      keypose_image.remove()
      keypose_image = ax2.imshow(avg_pose)

      stick_figure_image.remove()
      stick_figure_image = ax3.imshow(stick_figure)

      mvt_time.remove()
      mvt_time = ax5.axvline(x=frame['time'], ymin=0, ymax=4,color='r')

      interpose_time.remove()
      interpose_time = ax6.axvline(x=frame['time'], ymin=0, ymax=1,color='r')

      indiv_time.remove()
      indiv_time = ax8.axvline(x=frame['time'], ymin=0, ymax=3,color='r')

  print("SAVING FIGURE")

  with open(os.path.join(out_path, 'frame' + str(f).zfill(5) + '.png'), 'wb') as framefile:
    fig.savefig(framefile, format='png')

  print("CLEANING UP")
  plt.cla()
  plt.clf()
  plt.close('all')
  plt.close(fig)

  if f % 1000 == 0:
    gc.collect()
