fps, frames = get_video_stats('Jennie_SOLO_Lisa_Rhee.mp4')

CELL_HEIGHT=120

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

#reset_images_folder('/srv/choreo/solo_video')

plt.ioff()

XLIM=4090

START_FRAME = 7

coco_pts_labels = [coco_pts_short[x] for x in coco_pts_short]
margin = .1

maxwidth, maxheight = get_max_bbox(new_pose_data[:XLIM], figure_type='zeroified_figures', margin=.1)

p = 0
for i, frame in enumerate(new_pose_data):
  
  if i < START_FRAME:
    continue
    
  print('PROCESSING FRAME',i,"OF",len(new_pose_data)-1,"FRAMES")

  pose_fig = excerpt_pose('Jennie_SOLO_Lisa_Rhee.mp4', frame, 0, show=False, flip_figures=False, margin=0, width=maxwidth, height=maxheight, show_axis=False)
    
  lap_fig = excerpt_pose('Jennie_SOLO_Lisa_Rhee.mp4', frame, 0, show=False, flip_figures=False, plot_type="delaunay", margin=0, width=maxwidth, height=maxheight, show_axis=False)

  if pose_fig is None or lap_fig is None:
    print("POSE OR LAPLACIAN FIGURE IS EMPTY, SKIPPING")
    plt.rc('font', size=REALLY_BIG_SIZE)
    fig = plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT), constrained_layout=True)
    fig.dpi=DPI
    ax1 = fig.add_subplot(1,3,1)
    info_text = "FRAME " + str(i) + " " + str(round(frame['time'],2)) + "s\n" + "NO MOVEMENT DATA"
    ax1.set_anchor('NW')
    ax1.text(0, 1, info_text, horizontalalignment='left', verticalalignment='top')
    ax1.set_axis_off()
    
    framefile = os.path.join('/srv/choreo/solo_video', 'frame' + str(i).zfill(5) + '.png')
    #if not(os.path.isfile(framefile)):
    if True:
      im = fig2img(fig, FIG_WIDTH, FIG_HEIGHT)
      resized_im = im.resize((1048, 1026))
      # XXX ffmpeg doesn't like the PIL images, for some reason...
      resized_im.save(framefile, format='png', dpi=(72, 72))
      #fig.savefig(framefile, format='png', bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    plt.rc('font', size=BIGGER_SIZE)
    continue
    
  pose = fig2img(pose_fig, 4, 5)
  lap_pose = fig2img(lap_fig, 4, 5)

  cluster_pose_index = closest_matches[(i,p)]

  avg_pose = fig2img(plot_poses(cluster_avg_poses[cluster_pose_index], show_axis=False, show=False), 4, 6)
    
  heatmap = render_pose_distribution(heatmap, new_pose_data[:XLIM], labels, descriptors, closest_matches=closest_matches, video_file='Jennie_SOLO_Lisa_Rhee.mp4', time_index=frame['time'], show=False, xlim=XLIM)

  fig = plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))
    
  ax1 = fig.add_subplot(3,3,1)
  info_text = "FRAME " + str(i) + " " + str(round(frame['time'],2)) + "s"
  ax1.set_anchor('NW')
  ax1.imshow(pose)
  ax1.set_title(info_text, loc='center')
  ax1.set_axis_off()

  if np.isnan(smoothed_movement_series[0][i]):
    mvt_string = "0"
  else:
    mvt_string = str(round(smoothed_movement_series[0][i],2))
    
  ax2 = fig.add_subplot(3,3,2, xticks=np.arange(TOTAL_COORDS), yticks=np.arange(TOTAL_COORDS), xticklabels=coco_pts_labels, yticklabels=coco_pts_labels)
  plt.setp(ax2.get_xticklabels(), rotation=90) 
  ax2.text(0, -2.77, "MOVEMENT: " + mvt_string)
  ax2.set_anchor('NW')
  dmatrix = squareform(get_pose_matrix(frame,0))
  dmat = ax2.imshow(dmatrix, cmap='viridis', origin='upper')
  fig.colorbar(dmat, ax=ax2, shrink=0.8)

  ax3 = fig.add_subplot(3,3,3)
  ax3.set_title("DELAUNAY TRIANGULATION")
  ax3.set_anchor('NW')
  ax3.imshow(lap_pose)
  ax3.set_axis_off()

  ax4 = fig.add_subplot(3,3,4)
  ax4.set_title("DISTANCE MATRIX-BASED MOVEMENT", loc='left')
  ax4.set_anchor('SW')
  ax4.plot(frame_times,smoothed_movement_series[0],'b')
  ax4.axvline(x=frame['time'], ymin=0, ymax=1, color='r')
  ax4.margins(x=0)

  ax5 = fig.add_subplot(3,3,5)
  ax5.set_title("GRAPH LAPLACIAN-BASED MOVEMENT", loc='left')
  ax5.set_anchor('SE')
  ax5.plot(frame_times,laplacian_movement_series[0],'g')
  ax5.axvline(x=frame['time'], ymin=0, ymax=1, color='r')
  ax5.margins(x=0)

  ax6 = fig.add_subplot(3,3,6)
  ax6.set_title("MOST SIMILAR KEY POSE: " + str(cluster_pose_index))
  ax6.set_anchor('SW')
  ax6.imshow(avg_pose)
  ax6.set_axis_off()

  ax7 = fig.add_subplot(3,1,3)
  ax7.set_title("KEY POSES DISTRIBUTION")
  ax7.set_anchor('SW')
  ax7.plot(heatmap)
  im = ax7.imshow(heatmap, cmap='viridis_r')
  ax7.set_yticks(np.arange(CELL_HEIGHT/2, (len(label_keys)*CELL_HEIGHT)+CELL_HEIGHT/2, CELL_HEIGHT))
  ax7.set_yticklabels(np.arange(len(label_keys)))
  ax7.set_xticks(np.arange(0,len(new_pose_data[:XLIM]), fps*20))
  ax7.set_xticklabels([int(new_pose_data[k]['time']) for k in range(0,len(new_pose_data[:XLIM]),int(math.ceil(fps*20)))])
  heatmap_time = ax7.axvline(x=frame['time']*fps, ymin=0, ymax=len(label_keys)*CELL_HEIGHT,color='r')
      
  fig.dpi=DPI

  # Cheating way to get a margin at the top of the figure
  fig.suptitle('  ')

  with open(os.path.join('/srv/choreo/solo_video', 'frame' + str(i).zfill(5) + '.png'), 'wb') as framefile:
    fig.savefig(framefile, format='png', bbox_inches='tight', dpi=DPI)

  plt.cla()
  plt.clf()
  plt.close('all')
  plt.close(fig)
    
  if f % 1000 == 0:
    gc.collect()
