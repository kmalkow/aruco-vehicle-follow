# --------------------------------------------------------------------------
# Author:           Kevin Malkow
# Date:             02/07/23
# Affiliation:      TU Delft, IMAV 2023
#
# Version:          1.0 
# 
# Description:  
# Detect Aruco marker from images.
# 
# Upcoming Version: 2.0
# Detect Aruco marker from video stream.
#  -------------------------------------------------------------------------

import numpy as np
import os
import cv2
import cv2.aruco

# -------------- Load Image and Display Image --------------
#  Define image path
path = "/home/kevin/IMAV2023/Aruco_Marker_Pictures/ArucoMarker9.jpeg"  

# Read image
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(path)

# -------------- Aruco Marker Detection --------------
# Load dictionary for aruco marker
arucoDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

# Initiate aruco marker detection parameters
arucoParameters =  cv2.aruco.DetectorParameters()

# Aruco marker detection setup
arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)

# List of aruco marker detected corners, IDs corresponding to each aruco marker, and rejected aruco markers
(markerCorners, markerIDs, rejectedCandidates) = arucoDetector.detectMarkers(img)

# -------------- Draw Resulting Detection --------------
# Check if at least one aruco marker was detected
if len(markerCorners) > 0:
    # Iterate over aruco markers
    for i in range(0, len(markerIDs)):  
        # Draw around the correctly detected aruco markers
        cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIDs)  
        # Draw around the rejected candidates
        # cv2.aruco.drawDetectedMarkers(img, rejectedCandidates, borderColor=(100, 0, 240))
        cv2.aruco.drawDetectedMarkers(img, rejectedCandidates, borderColor=(100, 200, 255))

# Display the resulting image
cv2.imshow('image', img)

# Wait until the "0" key is pressed to close the image window
cv2.waitKey(0)

# --------------------------------------------------------
# UNCOMMENT TO SAVE IMAGE
# # Change the current directory to specified directory 
# os.chdir("/home/kevin/IMAV2023/Aruco_Marker_Pictures")
    
# # Saving the image
# cv2.imwrite('ArucoMarkerDetected9.jpeg', img)
# --------------------------------------------------------

# Remove image from memory
cv2.destroyAllWindows()

