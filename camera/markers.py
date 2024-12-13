import argparse
import sys

import numpy as np
import cv2


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

PIXELS_PER_MM = 3.7795

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
                    help="path to output image containing ArUCo tag")
    ap.add_argument("-i", "--id", type=int, required=True,
                    help="ID of ArUCo tag to generate")
    ap.add_argument("-t", "--type", type=str,
                    default="DICT_ARUCO_ORIGINAL",
                    help="type of ArUCo tag to generate")
    ap.add_argument("-b", "--border", type=int,
                    default=1,
                    help="border width of ArUco tag to generate (bits")
    ap.add_argument("-l", "--length", type=float,
                    default=25.0,
                    help="length of ArUco tag to generate (mm)")
    args = vars(ap.parse_args())

    # Verify that supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            args["type"]))
        sys.exit(0)

    # Load ArUCo dictionary
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])

    print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
        args["type"], args["id"]))

    length = int(args["length"] * PIXELS_PER_MM)
    tag = np.zeros((length, length, 1), dtype="uint8")
    cv2.aruco.drawMarker(aruco_dict, args["id"], length, tag, args["border"])
    # Save generated ArUCo tag and display on screen
    cv2.imwrite(args["output"], tag)
    cv2.imshow("ArUCo Tag", tag)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
