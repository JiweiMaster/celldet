from scipy.ndimage.filters import gaussian_filter
import scipy
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
import torch

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    import torch
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        # sigma = 10
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

def genDensityMap(xywh, imgSize=640):
    xywh = np.copy(xywh) if isinstance(xywh, torch.Tensor) else xywh
    xyxy = xywhn2xyxy(xywh,w=imgSize,h=imgSize)
    centerPoints = np.zeros((xyxy.shape[0], 2))
    centerPoints[..., 0] = (xyxy[..., 2] + xyxy[..., 0]) / 2
    centerPoints[..., 1] = (xyxy[..., 3] + xyxy[..., 1]) / 2
    centerPoints = np.asarray(centerPoints, np.uint16)
    # print(centerPoints)
    density_map_point = np.zeros((imgSize, imgSize), np.uint8)
    # print(centerPoints)
    density_map_point[centerPoints[..., 1], centerPoints[..., 0]] = 1
    density_map_point = gaussian_filter_density(density_map_point)
    return density_map_point



if __name__ == "__main__":
    bboxes = np.array([[0.428711,0.375977,0.0527344 ,0.0761719],
                    [0.739258 , 0.272461 , 0.0722656 ,0.0839844],
                    [0.755859 , 0.587891 , 0.0625   , 0.0546875],
                    [0.507812  ,0.293945 , 0.0625   , 0.0644531],
                    [0.478516 , 0.354492 , 0.0546875 ,0.0410156],
                    [0.537109 , 0.234375 , 0.0820312 ,0.0546875],
                    [0.365234 , 0.385742 , 0.0703125 ,0.0527344],
                    [0.31543  , 0.428711 , 0.0761719 ,0.0761719],
                    [0.454102 , 0.305664 , 0.0644531 ,0.0566406],
                    [0.584961 , 0.168945 , 0.0566406 ,0.0488281]])
    # 
    bboxes = torch.tensor(bboxes)
    out = genDensityMap(bboxes)
    plt.imsave('out.png', out)

