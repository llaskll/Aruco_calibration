import cv2
import numpy as np

markersX = 1
markersY = 1
markerLength = 102
markerSeparation = 20
margins = markerSeparation
borderBits = 10
showImage = True

width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins
height = markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
board = cv2.aruco.GridBoard_create(markersX, markersY, float(markerLength), float(markerSeparation), dictionary)
print(cv2.aruco_GridBoard.getGridSize(board))

img = cv2.aruco_GridBoard.draw(board, (width, height), 1)
cv2.imwrite('marker.png', img)
