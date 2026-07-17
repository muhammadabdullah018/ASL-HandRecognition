"""
The only file that talks to MediaPipe. Everything else consumes 63 floats.


"""
from __future__ import annotations

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63
WRIST, MIDDLE_MCP = 0, 9


# ------------------------------------------------------------------ features
def normalize_landmarks(pts: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """
    (21,3) aspect-corrected landmarks -> (63,) translation/scale invariant vector.

    Steps:
      mirror left->right  |  wrist to origin  |  scale by palm length
      optional: rotate so the palm points up
    """
    p = np.asarray(pts, dtype=np.float32).reshape(NUM_LANDMARKS, 3).copy()

    if handedness == "Left":
        p[:, 0] = -p[:, 0]

    # wrist becomes the origin -> kills position in frame
    p -= p[WRIST]

    # palm length (wrist -> middle finger knuckle) is a RIGID reference.
    # Don't use max-distance-from-wrist: that changes when fingers extend,
    # so the scale would depend on the letter you're signing.
    scale = float(np.linalg.norm(p[MIDDLE_MCP, :2]))
    p /= max(scale, 1e-6)

    if config.USE_ROTATION_NORM:
        # rotate about z so the palm vector points straight up (-y)
        vx, vy = p[MIDDLE_MCP, 0], p[MIDDLE_MCP, 1]
        theta = np.arctan2(vy, vx) + np.pi / 2.0
        c, s = np.cos(-theta), np.sin(-theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        p[:, :2] = p[:, :2] @ rot.T

    return p.reshape(-1)


def aspect_correct(landmark_list, frame_w: int, frame_h: int) -> np.ndarray:
    """MediaPipe NormalizedLandmarks -> (21,3) isotropic array."""
    ar = frame_w / frame_h
    return np.array(
        [[lm.x * ar, lm.y, lm.z * ar] for lm in landmark_list],
        dtype=np.float32,
    )


def palm_length(corrected: np.ndarray) -> float:
    """Wrist -> middle-MCP distance in aspect-corrected space. Same rigid
    reference normalize_landmarks() scales by - reused here as the "how big
    is this hand" unit for the two-hand gesture below."""
    return float(np.linalg.norm(corrected[MIDDLE_MCP, :2] - corrected[WRIST, :2]))


def hands_close(corrected_a: np.ndarray, corrected_b: np.ndarray, factor: float = 1.4) -> bool:
    """True if two hands' wrists are close relative to their own size - i.e.
    pressed/held together, not just both visible in frame. Comparing in
    aspect-corrected space (not raw pixels) keeps the threshold the same
    whether the hands are near or far from the camera. Used for the
    two-hands-together "space" gesture."""
    dist = float(np.linalg.norm(corrected_a[WRIST, :2] - corrected_b[WRIST, :2]))
    scale = (palm_length(corrected_a) + palm_length(corrected_b)) / 2
    return dist < factor * max(scale, 1e-6)


def is_fist(corrected: np.ndarray) -> bool:
    """True if all four fingers (index, middle, ring, pinky) are curled into a fist.
    Checked by seeing if fingertips are closer to the wrist than the MCP joints."""
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip_dist = float(np.linalg.norm(corrected[tip, :2] - corrected[WRIST, :2]))
        mcp_dist = float(np.linalg.norm(corrected[mcp, :2] - corrected[WRIST, :2]))
        if tip_dist > mcp_dist:
            return False
    return True


def hands_raised(px_a: np.ndarray, px_b: np.ndarray, frame_h: int, frac: float = 0.38) -> bool:
    """True if both hands' centers sit in the top `frac` of the frame - a
    raised-arms pose. Deliberately pixel/position-based, unlike hands_close():
    "up" is about where you are in the frame (gravity-relative), not the
    hand's own geometry, so aspect-corrected/scale-invariant space is the
    wrong space for this one. Used for the both-hands-up "celebrate" gesture."""
    y = (float(px_a[:, 1].mean()) + float(px_b[:, 1].mean())) / 2
    return y < frame_h * frac


# ------------------------------------------------------------------ detector
class HandTracker:
    """Thin wrapper over MediaPipe Tasks HandLandmarker in VIDEO mode."""

    def __init__(self, num_hands: int = 1):
        if not config.LANDMARKER_TASK.exists():
            raise FileNotFoundError(
                f"Missing {config.LANDMARKER_TASK}\nRun: python setup_model.py"
            )
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(config.LANDMARKER_TASK)
            ),
            running_mode=vision.RunningMode.VIDEO,  # uses inter-frame tracking
            num_hands=num_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONF,
            min_tracking_confidence=config.MIN_TRACKING_CONF,
        )
        self.detector = vision.HandLandmarker.create_from_options(opts)

    def process_all(self, rgb_frame: np.ndarray, timestamp_ms: int):
        """
        Like process(), but returns EVERY detected hand (up to num_hands),
        each as (px (21,2) int, feats (63,), handedness, corrected (21,3)).
        Needed for gestures that involve both hands at once - process()
        only ever looks at the first one.
        """
        h, w = rgb_frame.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self.detector.detect_for_video(mp_img, timestamp_ms)

        out = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            raw = hd[0].category_name
            handed = "Left" if raw == "Right" else "Right"
            corrected = aspect_correct(lms, w, h)
            feats = normalize_landmarks(corrected, handed)
            px = np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms], dtype=np.int32)
            out.append((px, feats, handed, corrected))
        return out

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int):
        """
        rgb_frame: ALREADY horizontally flipped (selfie view).
        returns (pixel_pts (21,2) int, feature_vec (63,), handedness) or None
        """
        hands = self.process_all(rgb_frame, timestamp_ms)
        if not hands:
            return None
        px, feats, handed, _corrected = hands[0]
        return px, feats, handed

    def close(self):
        self.detector.close()
