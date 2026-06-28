# Real-Time Colour Detection — Jetson Nano + RPi Camera
## Project Setup & Run Guide

---

## Hardware Required
- NVIDIA Jetson Nano Developer Kit (2GB or 4GB)
- Raspberry Pi Camera Module v2 (connected via CSI ribbon cable)
- Monitor / HDMI display (or SSH with X forwarding)

---

## 1. Enable the Raspberry Pi Camera

```bash
# Verify camera is detected
ls /dev/video*

# Quick test with nvgstcapture
nvgstcapture-1.0
```

If the camera is not detected, check the CSI ribbon cable connection and run:
```bash
sudo modprobe nvgstv4l2
```

---

## 2. Install Dependencies

```bash
# OpenCV (with GStreamer support — usually pre-installed on Jetson)
sudo apt-get update
sudo apt-get install python3-opencv -y

# Or install via pip if needed
pip3 install opencv-python numpy
```

> **Note:** The Jetson Nano JetPack image ships with OpenCV compiled with GStreamer support. Using the system OpenCV (`python3-opencv`) is recommended over pip to keep GStreamer support intact.

---

## 3. Run the Main Colour Detection Script

```bash
python3 color_detection_jetson.py
```

### Controls
| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit the program |
| `S` | Save a screenshot as `screenshot_XXX.jpg` |

---

## 4. Tune Custom Colours (HSV Calibration)

If you want to detect a new/custom colour, use the tuner:

```bash
python3 hsv_tuner.py
```

- Drag the `H_Low`, `S_Low`, `V_Low`, `H_High`, `S_High`, `V_High` trackbars
- Point the camera at your target colour
- Adjust until only the target colour appears white in the Mask pane
- Press `S` to print the HSV values to the terminal
- Add those values to the `COLOR_RANGES` dict in `color_detection_jetson.py`

---

## 5. Project File Structure

```
project/
├── color_detection_jetson.py   # Main real-time detection script
├── hsv_tuner.py                # Interactive HSV calibration tool
└── README.md                   # This file
```

---

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not open camera` | Check CSI cable; run `nvgstcapture-1.0` to verify |
| Low FPS | Reduce `display_width/height` in `gstreamer_pipeline()` |
| Wrong colours detected | Run `hsv_tuner.py` and recalibrate HSV ranges |
| Black screen | Try `flip_method=2` in `gstreamer_pipeline()` |
| OpenCV has no GStreamer | Use system OpenCV: `sudo apt install python3-opencv` |

---

## 7. Colours Detected (Default)

| Colour | Dual-mask? |
|--------|-----------|
| Red    | ✅ Yes (hue wrap) |
| Orange | No |
| Yellow | No |
| Green  | No |
| Cyan   | No |
| Blue   | No |
| Purple | No |
| Pink   | No |
| White  | No |
| Black  | No |

---

## 8. Key Concepts

- **HSV colour space** is used instead of BGR because it separates hue (colour type) from brightness, making colour detection robust to lighting changes.
- **GaussianBlur** before converting to HSV reduces noise and prevents spurious detections.
- **Morphological operations** (open + dilate) clean up the binary mask.
- **Red wraps around hue=180**, so two masks are OR'd together.
- **Coverage %** shown in each bounding box = area of detected region ÷ total frame area.
