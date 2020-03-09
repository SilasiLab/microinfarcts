import math
import numpy as np

rad = math.pi / 180.0
R = 6378137.0


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lon1,lat1) and (lon2,lat2)
    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the se*cond point
    param lon2: float, longitude of the second point
    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """

    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def s_lcss(t0, t1, eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float
    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1, 0], t0[i - 1, 1], t1[j - 1, 0], t1[j - 1, 1]) < eps:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    lcss = 1 - float(C[n0][n1]) / min([n0, n1])
    return lcss

def eucl_dist(x, y):
    """
    Usage
    -----
    L2-norm between point x and y
    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array
    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    dist = np.linalg.norm(x - y)
    return dist

def discret_frechet(t0, t1):
    """
    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q
    Parameters
    ----------
    param t0 : px2 numpy_array, Trajectory t0
    param t1 : qx2 numpy_array, Trajectory t1
    Returns
    -------
    frech : float, the discret frechet distance between trajectories t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = max(eucl_dist(t0[i - 1], t1[j - 1]), min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j]))
    dtw = C[n0, n1]
    return dtw

def relative_distance(t0: list, t1: list):
    assert len(t0) == len(t1)
    length = len(t0)
    distance1 = []
    distance2 = []
    for i in range(length - 1):
        for j in range(i + 1, length):
            distance1.append(eucl_dist(t0[i], t0[j]))
            distance2.append(eucl_dist(t1[i], t1[j]))

    similarity_distance = 0
    for i in range(length):
        similarity_distance += (distance1[i] - distance2[i]) ** 2.

    for i in range(length - 1):
        for j in range(i + 1, length):
            angle1 = (t0[i][0] - t0[j][0]) / (t0[i][1] - t0[j][1])
            angle2 = (t1[i][0] - t1[j][0]) / (t1[i][1] - t1[j][1])
            temp_distance = (angle1 - angle2) ** 2.
            similarity_distance += (temp_distance * 1e4)
    # print(similarity_distance)
    return similarity_distance
