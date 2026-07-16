"""
Step 0. Downloads the MediaPipe hand landmark model (~7 MB).

The Tasks API does NOT bundle weights - you supply the .task file yourself.
This is the part that trips people up when they migrate off mp.solutions.
"""
import sys
import urllib.request

import config


def main():
    if config.LANDMARKER_TASK.exists():
        mb = config.LANDMARKER_TASK.stat().st_size / 1e6
        print(f"already present: {config.LANDMARKER_TASK} ({mb:.1f} MB)")
        return

    print(f"downloading -> {config.LANDMARKER_TASK}")
    try:
        urllib.request.urlretrieve(config.LANDMARKER_URL, config.LANDMARKER_TASK)
    except Exception as e:
        print(f"FAILED: {e}\n\nDownload manually:\n  {config.LANDMARKER_URL}\nSave to: {config.LANDMARKER_TASK}")
        sys.exit(1)

    mb = config.LANDMARKER_TASK.stat().st_size / 1e6
    print(f"done ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
