# --------------------------------------------------------------------------
# Author:           Kevin Malkow and Sergio Marin Petersen
# Date:             11/09/23
# Affiliation:      TU Delft, IMAV 2023
#
# Version:          2.0 
# 
# Description:  
# - Detect Aruco markers and estimate Aruco marker position from real-time video stream
# - Get attitude, NED position, reference latitude, reference longitude, and reference altitude from drone through Ivybus
# - Convert Aruco marker position in image plane coordinates to NED coordinates for navigation algorithm
#   using drone attitude
# - Convert Aruco marker position to latitude, longitude, and altitude
# - Moving waypoint to Aruco marker position in latitude, longitude, and altitude 
#
#  -------------------------------------------------------------------------

                                        # LIBRARY DEFINITION #
# ------------------------------------------------------------------------------------------------------- #
# --------- General --------- # 
# import time
# import math
# import csv
# import sys
import cv2
# import pymap3d
import cv2.aruco
# import threading
# import numpy as np

                                        # STREAM WORKING CHECK #
# # ------------------------------------------------------------------------------------------------------- #
# # cap = cv2.VideoCapture(2)
# # cap = cv2.VideoCapture("rtsp://192.168.43.1:8554/fpv_stream") # Create a VideoCapture object (input is for herelink wifi connection)
# cap = cv2.VideoCapture("rtsp://192.168.42.129:8554/fpv_stream") # Create a VideoCapture object (input is for herelink bluetooth tethering)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
# 
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# 
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

                                      # LOAD CAMERA PARAMETERS #
# ------------------------------------------------------------------------------------------------------- #
pathLoad = './CameraCalibration_Variables/Videos/MAPIR_cameraCalibration_Video_w1920_h1080_HERELINKV2.xml'
cv_file = cv2.FileStorage(pathLoad, cv2.FILE_STORAGE_READ)
camera_Matrix = cv_file.getNode("cM").mat()
distortion_Coeff = cv_file.getNode("dist").mat()
cv_file.release()

                                        # VARIABLE DEFINITION #
# ------------------------------------------------------------------------------------------------------- #
MARKER_SIZE = 1.107                   # Size of Aruco marker in [m] -> 1.107 [m]||0.35 [m]

                                              # VIDEO #
# ------------------------------------------------------------------------------------------------------- #
# --------- Load Video --------- #
cap = cv2.VideoCapture("rtsp://192.168.42.129:8554/fpv_stream") # Create a VideoCapture object (input is for herelink bluetooth tethering)
FPS = cap.get(cv2.CAP_PROP_FPS)                                                          # Read FPS from input video
print(FPS)

# --------- Functioning? --------- #
if (cap.isOpened()== False):                                                             # Check if camera opened successfully
  print("Error: cannot open video file or stream")
 
# --------- Resolution --------- #
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print(frame_width)
print(frame_height)

# # --------- Write Video Setup --------- #
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')                                                     # Define video codec (FOURCC code)
out = cv2.VideoWriter('./Live_Videos/TEST0_CompleteV4.mp4', 
                      fourcc, FPS, (frame_width, frame_height))                                      # Create VideoWriter object 

                                    # ARUCO MARKER DETECTION SETUP #
# ------------------------------------------------------------------------------------------------------- #
# --------- Load Specific ID=700 Dictionary --------- # 
baseDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
arucoDictionary = cv2.aruco.Dictionary(baseDictionary.bytesList[700], 5, 6)

# --------- Set Detection Parameters --------- # 
arucoParameters =  cv2.aruco.DetectorParameters()

# STEP 1: Adaptive thresholding parameters
arucoParameters.adaptiveThreshWinSizeMin  = 3
arucoParameters.adaptiveThreshWinSizeMax  = 12
arucoParameters.adaptiveThreshWinSizeStep = 3
arucoParameters.adaptiveThreshConstant    = 11

# STEP 2: Contour filtering parameters
arucoParameters.polygonalApproxAccuracyRate = 0.04
arucoParameters.minDistanceToBorder         = 10

# STEP 3: Bit extraction parameters (large influence on detection performance, default = 4)
arucoParameters.perspectiveRemovePixelPerCell = 1

# STEP 4: Corner refinement -> Improves accuracy of Aruco marker pose estimation
arucoParameters.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParameters.cornerRefinementWinSize       = 7
arucoParameters.cornerRefinementMinAccuracy   = 0.1

# --------- Build Aruco Marker Detector --------- # 
arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)

# --------- Set Iteration Counter --------- # 
C_STEP = 0

                                            # RUN MAIN LOOP #
# ------------------------------------------------------------------------------------------------------- #
while(cap.isOpened()): 
  # --------- Read Frame-by-Frame --------- # 
  ret, frame = cap.read()

  if ret == True: # If frame read correctly            
    # --------- Convert to Grayscale --------- # 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------- Aruco Marker Detection --------- # 
    (markerCorners, _, _) = arucoDetector.detectMarkers(gray_frame)
    
    if len(markerCorners) > 0: # At least one marker detected
      # --------- Update Iteration Counter --------- # 
      C_STEP = C_STEP + 1

      # --------- Aruco Marker Pose Estimation --------- # 
      rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, MARKER_SIZE, camera_Matrix, distortion_Coeff)
      # (rvec - tvec).any()    # Remove Numpy value array error

      # if rvec is None:
      #   print(markerCorners)
      #   continue
      
      # if tvec is None:
      #   print(markerCorners)
      #   continue

      # --------- Save and Print X, Y, and Z --------- # 
      print(f"-------- ITERATION: {C_STEP} --------") 
      X_ARUCO = tvec[0][0][0]
      print(f"Aruco X: {X_ARUCO}")

      Y_ARUCO = tvec[0][0][1]
      print(f"Aruco Y: {Y_ARUCO}")

      Z_ARUCO = tvec[0][0][2]
      print(f"Aruco Z: {Z_ARUCO}")

    # --------- Write Video --------- # 
    out.write(frame)
    
    # --------- Display Output Frame --------- # 
    cv2.imshow('Frame', frame)

    # --------- Stop Code Execution (Press 'q') --------- # 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
            
  # --------- Break While Loop (No Frame Retrieved) --------- # 
  else:
    print('Error: frame not retrieved')  
    break

                                            # CLOSE CODE PROPERLY #
# ------------------------------------------------------------------------------------------------------- #
# --------- Release/Stop Objects --------- # 
cap.release()
out.release()

# --------- Close Frames --------- # 
cv2.destroyAllWindows()
