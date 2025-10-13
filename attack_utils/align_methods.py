import os
import cv2
import dlib
import numpy as np

# ArcFace 112×112 5-point template (left eye, right eye, nose, left mouth, right mouth)
ARC_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# Resolve the 81-landmark predictor in this repo
_PREDICTOR_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'shape_predictor_81_face_landmarks.dat'
))

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(_PREDICTOR_PATH)

def _detect_landmarks(image_bgr: np.ndarray) -> np.ndarray:
    # returns (N_landmarks, 2) float32
    dets = _detector(image_bgr, 1)
    if len(dets) == 0:
        # fallback: use full image box
        h, w = image_bgr.shape[:2]
        dets = [dlib.rectangle(0, 0, w, h)]
    shape = _predictor(image_bgr, dets[0])
    pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return pts  # 81 points (first 68 match the common indexing)

def _five_points_from_landmarks(pts: np.ndarray) -> np.ndarray:
    # Use 68-landmark indexing subset (present in the 81-model)
    # left eye: 36..41, right eye: 42..47, nose tip: 30, mouth corners: 48, 54
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    nose = pts[30]
    mouth_left = pts[48]
    mouth_right = pts[54]
    return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)

def _estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    # 2×3 similarity/affine matrix using 5 correspondences
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        # fallback: use first 3 points (eyes + nose)
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
    return M.astype(np.float32)

def align(image_bgr: np.ndarray, output_size=(112, 112)):
    # Input: H×W×C uint8 (BGR). Output: (aligned_bgr, M)
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, "Expected H×W×3 BGR image"
    pts = _detect_landmarks(image_bgr)
    src5 = _five_points_from_landmarks(pts)
    dst5 = ARC_TEMPLATE_112.copy()
    if output_size != (112, 112):
        sx = output_size[0] / 112.0
        sy = output_size[1] / 112.0
        dst5 = dst5 * np.array([sx, sy], dtype=np.float32)

    M = _estimate_affine(src5, dst5)
    aligned = cv2.warpAffine(image_bgr, M, (output_size[1], output_size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned.astype(np.uint8), M

def re_align(adv_img_bgr: np.ndarray, ali_img_bgr: np.ndarray, ori_img_bgr: np.ndarray, M: np.ndarray):
    # Warp adversarial aligned face back to original space using inverse M
    h, w = ori_img_bgr.shape[:2]
    Minv = cv2.invertAffineTransform(M.astype(np.float32))
    restored = cv2.warpAffine(adv_img_bgr, Minv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return restored.astype(np.uint8)