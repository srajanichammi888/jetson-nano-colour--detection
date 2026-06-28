"""
HSV Colour Tuner — Jetson Nano / RPi Camera
--------------------------------------------
Use this helper script to find the correct HSV range for any custom color.
Drag the trackbars to isolate your target color in the mask window.
Press  S  to print the final HSV values to the console.
Press  Q  to quit.
"""

import cv2
import numpy as np


def gstreamer_pipeline(
    capture_width=1280, capture_height=720,
    display_width=640, display_height=360,
    framerate=30, flip_method=0,
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
    )


def nothing(_):
    pass


def main():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[WARN] GStreamer pipeline failed. Trying default camera...")
        cap = cv2.VideoCapture(0)

    cv2.namedWindow("Tuner Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuner Controls", 400, 300)

    # Trackbars: lower and upper HSV bounds
    for name, val in [("H_Low", 0), ("S_Low", 0), ("V_Low", 0),
                      ("H_High", 180), ("S_High", 255), ("V_High", 255)]:
        cv2.createTrackbar(name, "Tuner Controls", val, 255 if name != "H_High" and "H" not in name else 180, nothing)

    # Fix H_High trackbar max
    cv2.setTrackbarMax("H_Low", "Tuner Controls", 180)
    cv2.setTrackbarMax("H_High", "Tuner Controls", 180)

    print("HSV Tuner started. Adjust trackbars and press S to print values.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_l = cv2.getTrackbarPos("H_Low",  "Tuner Controls")
        s_l = cv2.getTrackbarPos("S_Low",  "Tuner Controls")
        v_l = cv2.getTrackbarPos("V_Low",  "Tuner Controls")
        h_h = cv2.getTrackbarPos("H_High", "Tuner Controls")
        s_h = cv2.getTrackbarPos("S_High", "Tuner Controls")
        v_h = cv2.getTrackbarPos("V_High", "Tuner Controls")

        lower = np.array([h_l, s_l, v_l])
        upper = np.array([h_h, s_h, v_h])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Stack original | mask | result
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([
            cv2.resize(frame,   (320, 240)),
            cv2.resize(mask_3ch, (320, 240)),
            cv2.resize(result,  (320, 240)),
        ])
        cv2.putText(combined, "Original | Mask | Result",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("HSV Tuner", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print(f"\n  lower = np.array([{h_l}, {s_l}, {v_l}])")
            print(f"  upper = np.array([{h_h}, {s_h}, {v_h}])\n")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
