"""
python test_landmarks.py

Tests the feature maths without needing a camera or the .task model.
These invariances are the entire reason landmark features beat raw pixels:
if they don't hold, the model is learning your webcam, not your hand.
"""
import numpy as np

import config
from landmarks import (normalize_landmarks, aspect_correct, FEATURE_DIM,
                        hands_close, hands_raised, is_fist)


def fake_hand(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.2, (21, 3)).astype(np.float32)
    p[0] = [0.5, 0.8, 0.0]          # wrist
    p[9] = p[0] + [0.02, -0.15, 0]  # middle mcp above wrist
    return p


def test_shape():
    assert normalize_landmarks(fake_hand()).shape == (FEATURE_DIM,)


def test_translation_invariant():
    p = fake_hand()
    a = normalize_landmarks(p)
    b = normalize_landmarks(p + np.array([0.3, -0.2, 0.05], np.float32))
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()


def test_scale_invariant():
    """Hand closer to the camera must give the same vector."""
    p = fake_hand()
    a = normalize_landmarks(p)
    b = normalize_landmarks(p * 2.5)
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()


def test_wrist_at_origin():
    v = normalize_landmarks(fake_hand()).reshape(21, 3)
    assert np.allclose(v[0], 0, atol=1e-6)


def test_palm_length_is_one():
    v = normalize_landmarks(fake_hand()).reshape(21, 3)
    assert abs(np.linalg.norm(v[9, :2]) - 1.0) < 1e-4


def test_left_hand_mirrors_onto_right():
    p = fake_hand()
    mirrored = p.copy(); mirrored[:, 0] = -mirrored[:, 0]
    a = normalize_landmarks(p, "Right")
    b = normalize_landmarks(mirrored, "Left")
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()


def test_degenerate_hand_does_not_crash():
    """All points collapsed -> must not divide by zero."""
    v = normalize_landmarks(np.zeros((21, 3), np.float32))
    assert np.isfinite(v).all()


def test_rotation_norm_makes_it_tilt_invariant():
    old = config.USE_ROTATION_NORM
    config.USE_ROTATION_NORM = True
    try:
        p = fake_hand()
        th = np.deg2rad(35)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, -s], [s, c]], np.float32)
        q = p.copy(); q[:, :2] = q[:, :2] @ rot.T
        a = normalize_landmarks(p)
        b = normalize_landmarks(q)
        assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()
    finally:
        config.USE_ROTATION_NORM = old


def test_rotation_off_is_NOT_tilt_invariant():
    """Sanity: with the flag off, tilt must change the features."""
    assert config.USE_ROTATION_NORM is False
    p = fake_hand()
    th = np.deg2rad(35)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s], [s, c]], np.float32)
    q = p.copy(); q[:, :2] = q[:, :2] @ rot.T
    assert not np.allclose(normalize_landmarks(p), normalize_landmarks(q), atol=1e-3)


def test_aspect_correction_applied():
    class LM:
        def __init__(s, x, y, z): s.x, s.y, s.z = x, y, z
    lms = [LM(0.5, 0.5, 0.1)] * 21
    out = aspect_correct(lms, 1280, 720)
    assert abs(out[0, 0] - 0.5 * (1280 / 720)) < 1e-5
    assert abs(out[0, 1] - 0.5) < 1e-5


# ------------------------------------------------------------ gesture geometry
def test_hands_close_true_when_wrists_touch():
    a, b = fake_hand(1), fake_hand(2)
    b = b + (a[0] - b[0])  # translate b (whole hand, same internal scale) onto a's wrist
    assert hands_close(a, b)


def test_hands_close_false_when_far_apart():
    a, b = fake_hand(1), fake_hand(2)
    b = b + (a[0] - b[0]) + np.array([5.0, 0, 0], np.float32)  # miles past palm length
    assert not hands_close(a, b)


def test_hands_raised_true_near_top_of_frame():
    px_a = np.tile(np.array([100, 40]), (21, 1))
    px_b = np.tile(np.array([300, 55]), (21, 1))
    assert hands_raised(px_a, px_b, frame_h=720, frac=0.38)


def test_hands_raised_false_near_bottom_of_frame():
    px_a = np.tile(np.array([100, 650]), (21, 1))
    px_b = np.tile(np.array([300, 600]), (21, 1))
    assert not hands_raised(px_a, px_b, frame_h=720, frac=0.38)


def _hand_with_fingers(tip_dists):
    """21x3 hand centered at the wrist; only the 4 finger tip/MCP pairs
    is_fist() reads are meaningfully placed. tip_dists: {tip_idx: distance
    from wrist}, MCP is always fixed closer in so curled vs extended is
    just tip distance crossing that line."""
    p = np.zeros((21, 3), np.float32)
    mcp_dist = 0.3
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        p[mcp] = [mcp_dist, 0, 0]
        p[tip] = [tip_dists[tip], 0, 0]
    return p


def test_is_fist_true_when_all_fingers_curled():
    p = _hand_with_fingers({8: 0.1, 12: 0.1, 16: 0.1, 20: 0.1})
    assert is_fist(p)


def test_is_fist_false_when_fingers_extended():
    p = _hand_with_fingers({8: 0.9, 12: 0.9, 16: 0.9, 20: 0.9})
    assert not is_fist(p)


def test_is_fist_false_if_only_one_finger_extended():
    p = _hand_with_fingers({8: 0.1, 12: 0.1, 16: 0.1, 20: 0.9})
    assert not is_fist(p)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
