
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
for d in (DATA_DIR, MODEL_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

# ---------------------------------------------------------------- classes
# J and Z are excluded on purpose: both require MOTION.
# A single-frame classifier physically cannot represent them.
# This is a scope decision, not a bug. Say so in the post.
LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # 24 static letters
assert len(LETTERS) == 24

# ---------------------------------------------------------------- files
LANDMARKER_TASK = MODEL_DIR / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATASET_CSV = DATA_DIR / "landmarks.csv"
CLASSIFIER_PKL = MODEL_DIR / "classifier.pkl"

# ---------------------------------------------------------------- capture
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
SAMPLES_PER_LETTER = 150
COUNTDOWN_SEC = 3

# ---------------------------------------------------------------- features
# Rotation normalization: rotates the hand so wrist->middle-MCP points "up".
# Makes the model tilt-invariant. Sometimes helps, sometimes hurts (some ASL
# letters differ mainly by orientation). Train both ways, keep the winner,
# and report the delta. That comparison IS your post content.
USE_ROTATION_NORM = False

# ---------------------------------------------------------------- inference
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5
CONF_THRESHOLD = 0.80      # below this, predict nothing rather than guess

# How long a letter must be held before it commits.
# You asked for 3.0. Try it, but I'd start at 1.5 - see the note in README.
# Press [ and ] in spell_mode.py to change this live, on camera, and decide
# with your own eyes instead of guessing at a config file.
HOLD_SECONDS = 1.5

SMOOTHING_WINDOW = 5       # majority vote over last N frames

# Both-hands-fist gesture -> space. Shorter than HOLD_SECONDS on
# purpose: it's a deliberate two-hand action, not a pose you'd drift into by
# accident the way you might rest one hand mid-letter.
SPACE_HOLD_SECONDS = 1.5
HANDS_CLOSE_FACTOR = 1.4   # wrist-distance threshold, in units of palm length

# Both-hands-up gesture -> big celebratory reveal of the spelled word.
# Longer hold than space: it's a deliberate "I'm done" moment, not something
# you want firing because your hands passed through a raised pose mid-sign.
CELEBRATE_HOLD_SECONDS = 1.0
CELEBRATE_SHOW_SECONDS = 2.5
HANDS_UP_FRAC = 0.38       # top fraction of the frame that counts as "up"

# ---------------------------------------------------------------- UI
# OpenCV is BGR, not RGB. Get this backwards and "systemBlue" renders orange.
# Palette is Apple's dark-mode system colors (Human Interface Guidelines),
# so it reads as native rather than "webcam demo": near-black elevated
# surfaces, a restrained gray text hierarchy, and exactly one accent hue
# doing double duty as the in-progress state. Green/red stay reserved for
# their one meaning each - confirmed / rejected - never used decoratively.
COL_BG = (30, 28, 28)            # systemBackground (dark, elevated)
COL_PANEL = (46, 44, 44)         # secondarySystemBackground - card fill
COL_FG = (247, 245, 245)         # label - primary text
COL_DIM = (157, 152, 152)        # secondaryLabel
COL_FAINT = (102, 99, 99)        # tertiaryLabel - quietest text, idle dots
COL_BORDER = (58, 56, 56)        # separator - 1px hairlines
COL_ACCENT = (255, 132, 10)      # systemBlue (dark) - in-progress / holding
COL_GOOD = (88, 209, 48)         # systemGreen (dark) - locked / confirmed
COL_BAD = (58, 69, 255)          # systemRed (dark) - no hand / rejected
COL_BONE = (235, 232, 230)       # skeleton idle - soft white, visionOS-style
COL_JOINT = (255, 255, 255)

# MediaPipe hand skeleton edges (21 landmarks)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]
