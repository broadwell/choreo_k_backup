 For each pose, compute its L1-normed inter-keypoint distance matrix. Then,
# to compare any two poses, we can measure the degree of correlation between
# their distance matrices via a statistical test, such as the Mantel test.
# XXX It's not obvious that normalizing the matrix really makes a difference to
# the final correlation comparison, but it doesn't seem to hurt, either...
# Note that if the pose representation has 17 keypoints, then each pose instance
# can be represented by a condensed distance matrix (or vector) of 136 elements.

# XXX Incorporate joint scores (probabilities) somehow? Perhaps using
# https://ecopy.readthedocs.io/en/latest/matrix_comps.html

from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import mantel
from sklearn.preprocessing import normalize

def matrixify_pose(coords_and_confidence):
  if coords_and_confidence.shape[0] == 0:
    return None
  coords = coords_and_confidence[:,:2]
  condensed_distance_matrix = normalize(pdist(coords, 'sqeuclidean').reshape(1, -1))[0,:]
  return condensed_distance_matrix

def get_pose_matrix(frame, figure_index=0, figure_type='flipped_figures'):
  if figure_type not in frame or figure_index > len(frame[figure_type])-1 or frame[figure_type][figure_index].data.shape[0] == 0:
    return None
  coords_and_confidence = frame[figure_type][figure_index].data
  return matrixify_pose(coords_and_confidence)


# For each pose, compute the Delaunay triangulation between its keypoints, then
# use the connections to build an adjacency matrix, which is then converted
# to its (normalized) Laplacian matrix (a single matrix that encapsulates the
# degree of each node and the connections between the nodes). Then you can
# subtract a pose's Laplacian from another's to get a measure of the degree of
# similarity or difference between them.

from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy import triu
import networkx as nx


def plot_delaunay(figure, image=None, show=True, show_axis=True):

  if hasattr(figure, 'data'):
    vis_figures = [figure]
  else:
    vis_figures = []
    for pose in figures:
      if pose.data.shape[0] == 0:
        continue
      vis_figures.append(pose)

  for figure in vis_figures:
    all_points = figure.data

    # For visualization, remove all [x,y,0] (unknown) coordinates.
    nonzero = (all_points!=0).all(axis=1)
    nz_points = all_points[nonzero]
    points = nz_points[:,:2]
    total_points = len(points)
    tri = Delaunay(points)

    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    plt.plot(points[:,0], points[:,1], 'o')
    
  if image is not None:
    plt.gca().imshow(image)
  else:
    plt.gca().set_aspect('equal')
  if not show_axis:
    plt.gca().set_axis_off()
  if show:
    plt.show()
    
  return plt.gcf()
    

def get_laplacian_matrix(frame, normalized=True, show=False, figure_index=0, figure_type='flipped_figures'):

  if figure_type not in frame or figure_index > len(frame[figure_type])-1 or frame[figure_type][figure_index].data.shape[0] == 0:
    return None
    
  all_points = frame[figure_type][figure_index].data

  # For visualization, remove all [x,y,0] (unknown) coordinates.
  nonzero = (all_points!=0).all(axis=1)
  nz_points = all_points[nonzero]
  points = nz_points[:,:2]
  total_points = len(points)
  tri = Delaunay(points)

  if show:
    plot_delaunay(frame[figure_type][figure_index])

  adjacency_matrix = lil_matrix((total_points, total_points), dtype=int)
  for i in np.arange(0, np.shape(tri.simplices)[0]):
    for j in tri.simplices[i]:
      if j < total_points:
        adjacency_matrix[j, tri.simplices[i][tri.simplices[i] < total_points]] = 1

  adjacency_graph = nx.from_scipy_sparse_matrix(adjacency_matrix)

  if normalized:
    lm = nx.normalized_laplacian_matrix(adjacency_graph)
  else:
    lm = nx.laplacian_matrix(adjacency_graph)

  return lm


# Laplacian tests

def compare_laplacians(p1, p2, figure_index=0, figure_type='flipped_figures', show=False):
  lm1 = get_laplacian_matrix(p1, figure_index=figure_index, figure_type=figure_type, show=show).todense()
  lm2 = get_laplacian_matrix(p2, figure_index=figure_index, figure_type=figure_type, show=show).todense()
  if lm1.shape[0] != lm2.shape[0]:
    return None
  movement = np.subtract(lm1, lm2)
  return 1 - abs(movement.sum())


