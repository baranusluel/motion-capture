# Basic 2D motion capture system based on object tracking with user-defined ROIs
# and manually calibrated 2D positioning.
# References:
#  Object tracking:
#  - https://www.pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/
#  - https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

from imutils.video import FPS
import argparse
import imutils
import time
import cv2

print("Run command with -h to see usage help")
# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="0",
        help="path to input video file or webcam ID")
ap.add_argument("-t", "--tracker", type=str, default="medianflow",
        help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize dictionary of strings to corresponding tracker implementations
OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
}
# grab specified tracker
trackers = cv2.MultiTracker_create()

print("Loading video...")

# initialize bounding box of object to track
initBox = None
# open webcam / specified video file
source = args["video"]
vs = cv2.VideoCapture(int(source) if source.isdigit() else source)
# initialize fps
fps = None

print("- Press C to calibrate coordinates")
print("- Press S to add an object ROI")
print("- Press Q to quit")

# Pixels to ground-truth calibration constants. By default assuming 1:1 mapping
calibrationRef = {"refPoint":
                    {"pixels": (0,0), "truth": (0,0)},
                "pixelsToTruthRatio": (1,1)}

def calibrateXY(pt, ref):
    x = (pt[0] - calibrationRef["refPoint"]["pixels"][0]) * \
        calibrationRef["pixelsToTruthRatio"][0] + \
        calibrationRef["refPoint"]["truth"][0]
    y = (pt[1] - calibrationRef["refPoint"]["pixels"][1]) * \
        calibrationRef["pixelsToTruthRatio"][1] + \
        calibrationRef["refPoint"]["truth"][1]
    return (x, y)

# video loop
while True:
    # get frame, check if reached EOS
    ret, frame = vs.read()
    if frame is None:
        break
    # resize frame and get dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    # check if currently tracking object
    if initBox is not None:
        # grab new bounding box
        (success, boxes) = trackers.update(frame)

        print("\n")
        realPoints = []

        # loop over bounding boxes and draw on frame
        for i in range(len(boxes)):
            box = boxes[i]
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 150, 0), 2)
            boxCenter = (x + w//2, y + h//2)
            cv2.circle(frame, boxCenter, 2, (0, 255, 0), 2)
            realXY = calibrateXY(boxCenter, calibrationRef)
            realPoints.append(realXY)
            print("Object %d, (%.4f, %.4f)" % (i, realXY[0], realXY[1]))
            idTextMargin = 4
            cv2.putText(frame, str(i), (x + w + idTextMargin, y + h + idTextMargin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for j in range(len(realPoints)-1):
                print("Distance %d to %d, (%.4f, %.4f)" %
                        (j, i, realXY[0] - realPoints[j][0], realXY[1] - realPoints[j][1]))

        # update FPS counter
        fps.update()
        fps.stop()

        # initialize set of info to display
        info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over info tuples and draw on frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if s key pressed, select bounding box
    if key == ord('s'):
        # select bounding box
        initBox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        # create tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, initBox)
        # start FPS tracker
        fps = FPS().start()

    # if c key pressed select calibration box
    elif key == ord('c'):
        # select bounding box
        calBox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
        (x, y, w, h) = [int(v) for v in calBox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        textMargin = 4

        # get x coordinate of top left corner
        frameCopy = frame.copy()
        cv2.putText(frameCopy, "X_0", (x - textMargin, y - textMargin),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Frame", frameCopy)
        cv2.waitKey(1)
        x_calib0 = float(input("Enter x coordinate of marked corner: "))

        # get y coordinate of top left corner
        frameCopy = frame.copy()
        cv2.putText(frameCopy, "Y_0", (x - textMargin, y - textMargin),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Frame", frameCopy)
        cv2.waitKey(1)
        y_calib0 = float(input("Enter y coordinate of marked corner: "))

        calibrationRef["refPoint"]["pixels"] = (x, y)
        calibrationRef["refPoint"]["truth"] = (x_calib0, y_calib0)

        # get x coordinate of bottom right corner
        frameCopy = frame.copy()
        cv2.putText(frameCopy, "X_1", (x + w + textMargin, y + h + textMargin),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Frame", frameCopy)
        cv2.waitKey(1)
        x_calib1 = float(input("Enter x coordinate of marked corner: "))

        # get y coordinate of bottom right corner
        frameCopy = frame.copy()
        cv2.putText(frameCopy, "Y_1", (x + w + textMargin, y + h + textMargin),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Frame", frameCopy)
        cv2.waitKey(1)
        y_calib1 = float(input("Enter y coordinate of marked corner: "))

        calibrationRef["pixelsToTruthRatio"] = ((x_calib1 - x_calib0) / w,
                                                (y_calib1 - y_calib0) / h)

        fps = FPS().start()

    # if q key pressed exit loop
    elif key == ord('q'):
        break

# cleanup
vs.release()
cv2.destroyAllWindows()
