# --------------------------------------------------------------------------
# Author:           Kevin Malkow
# Date:             05/07/23
# Affiliation:      TU Delft, IMAV 2023
#
# Version:          3.0 
# 
# Description:  
# Detect AND track Aruco marker from images -> requires camera calibration step ("IMAV2023_CameraCalibration.py")
# 
# Upcoming Version: 4.0
# Detect AND track Aruco marker from pre-recorded videos.
# 
# Upcoming Version: 5.0
# Detect AND track Aruco marker from video live-stream.
#  -------------------------------------------------------------------------

#  ------------------------------------------------------------------------- #
#                            LIBRARY DEFINITION                              #
#  ------------------------------------------------------------------------- #
import numpy as np
import glob
import cv2
import cv2.aruco
from pathlib import Path

#  ------------------------------------------------------------------------- #
#                            CONSTANT DEFINITION                             #
#  ------------------------------------------------------------------------- #
# Size of Aruco marker in [m]
MARKER_SIZE = 0.35

#  ------------------------------------------------------------------------- #
#                                 FUNCTIONS                                  #
#  ------------------------------------------------------------------------- #
def img_rescale(img, scale_percent):
# Image downscaling -> USED AS IMAGE INPUT AND AFFECTS PERFORMANCE OF DETECTOR
  scale_percent = 30 # Percent of original size
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#  ------------------------------------------------------------------------- #
#                           LOAD CAMERA VARIABLES                            #
#  ------------------------------------------------------------------------- #
pathLoad = '/home/kevin/IMAV2023/CameraCalibration_Variables/cameraCalibration.xml'   
cv_file = cv2.FileStorage(pathLoad, cv2.FILE_STORAGE_READ)
camera_Matrix = cv_file.getNode("cM").mat()
distortion_Coeff = cv_file.getNode("dist").mat()

cv_file.release()

#  ------------------------------------------------------------------------- #
#                           LOAD AND READ IMAGES                             #
#  ------------------------------------------------------------------------- #
#  Define image path
path = '/home/kevin/IMAV2023/Aruco_Marker_Data/04_07_2023/Pictures/Storey_2/*.JPG'   

# Initialise counter for formatting saved images
counter = 1

# Read images in a batch
for file in glob.iglob(path): 
    # Read image
    img = cv2.imread(file)

    #  ------------------------------------------------------------------------- #
    #                         ARUCO MARKER DETECTION                             #
    #  ------------------------------------------------------------------------- #    
    # Load dictionary for aruco marker
    arucoDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

    # Initiate aruco marker detection parameters
    arucoParameters =  cv2.aruco.DetectorParameters()

    # Aruco marker detection setup
    arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)

    # List of aruco marker detected corners, IDs corresponding to each aruco marker, and rejected aruco markers
    (markerCorners, markerIDs, rejectedCandidates) = arucoDetector.detectMarkers(img)

    # AT LEAST ONE MARKER DETECTED   
    if len(markerCorners) > 0:        
        # Failsafe -> Only consider Marker ID=700 (to nullify false positives)
        if any(x == 700 for x in markerIDs):
            # Iterate over aruco markers (Allow only 1 marker detection aince we know only ID=700 should be detected)
            totalMarkers = min(1, len(markerIDs))
            for i in range(0, totalMarkers):  
                #  ------------------------------------------------------------------------- #
                #                       ARUCO MARKER POSE ESTIMATION                         #
                #  ------------------------------------------------------------------------- #
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners[i], MARKER_SIZE, camera_Matrix, distortion_Coeff)

                # Remove Numpy value array error
                (rvec - tvec).any()  
                
                #  ------------------------------------------------------------------------- #
                #             COMPUTE AND SHOW EUCLIDEAN DISTANCE, X, Y, AND Z               #
                #  ------------------------------------------------------------------------- #
                # Compute Euclidean distance between two points in space -> sqrt(x^2 + y^2 + z^2)   
                euclideanDist = np.sqrt(tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2 + tvec[i][0][2] ** 2)
                
                # Print Euclidean distance, X, Y, and Z 
                print(f"Euclidean Distance: {euclideanDist}")
                print(f"X: {tvec[i][0][0]}")
                print(f"Y: {tvec[i][0][1]}")
                print(f"Z: {tvec[i][0][2]}")   
                
                #  ------------------------------------------------------------------------- #
                #                DRAW MARKERS, AXES, AND ADD TEXT TO IMAGES                  #
                #  ------------------------------------------------------------------------- #
                # Draw around the correctly detected aruco markers
                cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIDs)  

                # Draw axis on Aruco markers -> X = red, Y = green, Z = blue
                cv2.drawFrameAxes(img, camera_Matrix, distortion_Coeff, rvec, tvec, MARKER_SIZE/2) 

                # Add text to images 
                org_1 = (int(markerCorners[i][0][0][0]), int(markerCorners[i][0][0][1])) # origin
                text_1 = f"Dist: {round(euclideanDist, 2)}[m]"
                font_1 = cv2.FONT_HERSHEY_PLAIN
                fontScale_1 = 1.3
                color_1 = (0, 0, 255)
                lineThickness_1 = 2

                org_2 = (int(markerCorners[i][0][3][0]), int(markerCorners[i][0][3][1])) # origin
                text_2 = f"X: {round(tvec[i][0][0], 1)}[m] Y: {round(tvec[i][0][1], 1)}[m]"
                font_2 = cv2.FONT_HERSHEY_PLAIN
                fontScale_2 = 1.0
                color_2 = (0, 0, 255)
                lineThickness_2 = 2
                
                cv2.putText(img, text_1, org_1, font_1, fontScale_1, color_1, lineThickness_1, cv2.LINE_AA)
                cv2.putText(img, text_2, org_2, font_2, fontScale_2, color_2, lineThickness_2, cv2.LINE_AA)

                #  ------------------------------------------------------------------------- #
                #                         DISPLAY AND SAVE IMAGES                            #
                #  ------------------------------------------------------------------------- #
                # Increase resolution (very computationally heavy -> ONLY FOR DEMONSTRATION PURPOSES!!!)
                # img = cv2.pyrUp(img)

                # Resize image to fit
                img = cv2.resize(img, (960, 540))

                # Display the resulting image
                cv2.imshow('image', img)

                # Wait until the "0" key is pressed/wait 1 [ms] to close the image window
                cv2.waitKey(0)

                # # --------------------------------------------------------
                # # UNCOMMENT TO SAVE IMAGE            
                # # Saving the image
                # cv2.imwrite('/home/kevin/IMAV2023/Aruco_Marker_Data/04_07_2023/Pictures/Storey_2/POSE2_ArucoMarkerDetected_{}.JPG'.format(counter), img)
                # # --------------------------------------------------------

                # Remove image from memory
                cv2.destroyAllWindows()
        
    # Increment counter for formatting saved images
    counter += 1
