from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_ious(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute Intersection over Union (IoU) between two sets of boxes [x, y, w, h]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1_x1, b1_y1 = boxes1[:, 0], boxes1[:, 1]
    b1_x2, b1_y2 = boxes1[:, 0] + boxes1[:, 2], boxes1[:, 1] + boxes1[:, 3]
    b1_area = boxes1[:, 2] * boxes1[:, 3]

    b2_x1, b2_y1 = boxes2[:, 0], boxes2[:, 1]
    b2_x2, b2_y2 = boxes2[:, 0] + boxes2[:, 2], boxes2[:, 1] + boxes2[:, 3]
    b2_area = boxes2[:, 2] * boxes2[:, 3]

    ious = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    for i in range(len(boxes1)):
        inter_x1 = np.maximum(b1_x1[i], b2_x1)
        inter_y1 = np.maximum(b1_y1[i], b2_y1)
        inter_x2 = np.minimum(b1_x2[i], b2_x2)
        inter_y2 = np.minimum(b1_y2[i], b2_y2)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = b1_area[i] + b2_area - inter_area
        valid = union_area > 0
        ious[i, valid] = inter_area[valid] / union_area[valid]

    return ious


def iou_cost_matrix(tracks_boxes: np.ndarray, detections_boxes: np.ndarray) -> np.ndarray:
    """IoU cost matrix = 1.0 - IoU."""
    ious = bbox_ious(tracks_boxes, detections_boxes)
    return 1.0 - ious


def feature_cosine_distance(feats1: np.ndarray, feats2: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix between appearance feature vectors."""
    if len(feats1) == 0 or len(feats2) == 0:
        return np.ones((len(feats1), len(feats2)), dtype=np.float32)
    # Normed cosine similarity
    sim = np.dot(feats1, feats2.T)
    return np.clip(1.0 - sim, 0.0, 2.0)


def min_cost_matching(
    cost_matrix: np.ndarray, max_cost_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve optimal linear sum assignment (Hungarian algorithm) with threshold gating."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = list(set(range(cost_matrix.shape[0])) - set(row_ind))
    unmatched_cols = list(set(range(cost_matrix.shape[1])) - set(col_ind))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_cost_threshold:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
        else:
            matches.append((r, c))

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)
