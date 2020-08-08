
# Setup

def initialize_detector()
  try:
    device = torch.device('cuda')  # if cuda is available
  except:
    device = torch.device('cpu')

  net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k30w-200510-104256-cif-caf-caf25-o10s-0b5ba06f.pkl', download_progress=False)
  net = net_cpu.to(device)

  openpifpaf.decoder.CifSeeds.threshold = 0.5
  openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
  openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
  processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)

  preprocess = openpifpaf.transforms.Compose([
    openpifpaf.transforms.NormalizeAnnotations(),
    openpifpaf.transforms.CenterPadTight(16),
    openpifpaf.transforms.EVAL_TRANSFORM,
  ])


def detect_one_or_more_images(batch, processor):
  data = openpifpaf.datasets.PilImageList(batch, preprocess=preprocess)
  batch_size = len(batch)

  loader = torch.utils.data.DataLoader(
  data, batch_size=batch_size, pin_memory=True, 
  collate_fn=openpifpaf.datasets.collate_images_anns_meta)

  for images_batch, _, __ in loader:
    detections = processor.batch(net, images_batch, device=device)[0]
  
  return detections


def process_raw_image(image_path):
  pil_image = PIL.Image.open(image_path)
  detections = detect_one_or_more_images([pil_image], processor)
  #flipped_detections = flip_detections(detections)
  #rectified_detections = flip_detections(detections, rectify_x=True)
  #detections[0].text = "1"
  #new_detection = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON).set(detections[0].data, fixed_score=.87, category=3)

  #plot_poses(detections[0], pil_image, show=True)

  return detections
