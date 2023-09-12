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
import signal
# import pymap3d
import cv2.aruco
# import threading
import numpy as np

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

scale_percent_720p = 66.7
scaling_factor_X_720p = -0.305            # Scaling factor to account for reduced frame size in Aruco marker X measurements
scaling_factor_Y_720p = -0.2519           # Scaling factor to account for reduced frame size in Aruco marker Y measurements
scaling_factor_Z_720p = -0.101            # Scaling factor to account for reduced frame size in Aruco marker Z measurements

# rvec = np.zeros([1, 3])
# tvec = np.zeros([1, 3])

                                # FUNCTION -> VISUALISE LEGEND #
# ------------------------------------------------------------------------------------------------------- #
def timeout(timeout_duration=3):
  class TimeoutError(Exception):
    pass

  def handler(A, B):
    raise TimeoutError()
  
  # --------- Set Timeout Handler --------- #
  signal.signal(signal.SIGALRM, handler)
  signal.alarm(timeout_duration)

  try:
    # --------- Read Frame-by-Frame --------- # 
    print("STEP 5 -> Start receiving frame")
    ret, frame = cap.read()
    print("STEP 5 -> Grabbed frame")

  except TimeoutError as exc:
    print(exc)
    ret = False
    frame = None

  finally:
    signal.alarm(0)

  return ret, frame 

                                              # VIDEO #
# ------------------------------------------------------------------------------------------------------- #
# --------- Load Video --------- #
print("STEP 1 -> Starting videoCapture object")
cap = cv2.VideoCapture("rtsp://192.168.42.129:8554/fpv_stream") # Create a VideoCapture object (input is for herelink bluetooth tethering)
# cap = cv2.VideoCapture("rtsp://192.168.43.1:8554/fpv_stream") # Create a VideoCapture object (input is for herelink wifi connection)
print("STEP 1 -> Finished videoCapture object")
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
print("STEP 2 -> Started videoWriter object")
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')                                                     # Define video codec (FOURCC code)
out = cv2.VideoWriter('./Live_Videos/TEST0_CompleteV4.mp4', 
                      fourcc, FPS, (frame_width, frame_height))                                      # Create VideoWriter object 
print("STEP 2 -> Finished videoWrite object")

                                    # ARUCO MARKER DETECTION SETUP #
# ------------------------------------------------------------------------------------------------------- #
# --------- Load Specific ID=700 Dictionary --------- # 
baseDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
arucoDictionary = cv2.aruco.Dictionary(baseDictionary.bytesList[700], 5, 6)

print("STEP 3 -> Start aruco parameters")
# --------- Set Detection Parameters --------- # 
arucoParameters =  cv2.aruco.DetectorParameters()
print("STEP 3 -> Finished aruco parameters")
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
print("STEP 4 -> Setup arucoDetector")
arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)
print("STEP 4 -> Finished setup arucoDetector")

# --------- Set Iteration Counter --------- # 
C_STEP = 0

                                            # RUN MAIN LOOP #
# ------------------------------------------------------------------------------------------------------- #
while(cap.isOpened()): 

  # ret, frame = timeout() 
  print("STEP 5 -> Start receiving frame")
  ret, frame = cap.read()
  print("STEP 5 -> Grabbed frame")

  if ret == True: # If frame read correctly            
    print("STEP 6 -> ret = True")
    # --------- Convert to Grayscale --------- # 
    print("STEP 7 -> Start gray conversion")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("STEP 7 -> Finished gray conversion")
    # --------- Aruco Marker Detection --------- # 
    print("STEP 8 -> Start aruco detection")
    (markerCorners, _, _) = arucoDetector.detectMarkers(gray_frame)
    print("STEP 8 -> Finished aruco detection")
    
    # --------- Update Iteration Counter --------- # 
    C_STEP = C_STEP + 1

    print(f"-------- ITERATION: {C_STEP} --------") 

    if len(markerCorners) > 0: # At least one marker detected
      print("STEP 9 -> Aruco marker detected")
      # --------- Aruco Marker Pose Estimation --------- # 
      print("STEP 10 -> Start Aruco marker pose estimation")
      rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, MARKER_SIZE, camera_Matrix, distortion_Coeff)
      print("STEP 10 -> Retrieved rvec and tvec from Aruco marker pose estimation")
      (rvec - tvec).any()    # Remove Numpy value array error
      print("STEP 11 -> Rvec - tvec .any()")
      # if rvec is None:
      #   print(markerCorners)
      #   continue
      
      # if tvec is None:
      #   print(markerCorners)
      #   continue

      # --------- Save and Print X, Y, and Z --------- # 
      # print(f"-------- ITERATION: {C_STEP} --------") 
      X_ARUCO = tvec[0][0][0]
      # X_ARUCO = X_ARUCO*((scale_percent_720p/100) + scaling_factor_X_720p)
      print(f"Aruco X: {X_ARUCO}")

      Y_ARUCO = tvec[0][0][1]
      # Y_ARUCO = Y_ARUCO*((scale_percent_720p/100) + scaling_factor_Y_720p)      
      print(f"Aruco Y: {Y_ARUCO}")

      Z_ARUCO = tvec[0][0][2]
      # Z_ARUCO = Z_ARUCO*((scale_percent_720p/100) + scaling_factor_Z_720p)
      print(f"Aruco Z: {Z_ARUCO}")

    # --------- Write Video --------- # 
    print("STEP 12 -> Start video out.write")
    out.write(frame)
    print("STEP 12 -> Finish video out.write")

    # --------- Display Output Frame --------- # 
    cv2.imshow('Frame', frame)

    # --------- Stop Code Execution (Press 'q') --------- # 
    print("STEP 13 -> Quit code")
    if cv2.waitKey(1) & 0xFF == ord('q'):
     print("STEP 14 -> Code quit")
     break
            
  # --------- Break While Loop (No Frame Retrieved) --------- # 
  else:
    print('Error: frame not retrieved')  
    break

                                            # CLOSE CODE PROPERLY #
# ------------------------------------------------------------------------------------------------------- #
# --------- Release/Stop Objects --------- # 
print("STEP 15 -> Release/stop objects")
cap.release()
out.release()

# --------- Close Frames --------- # 
cv2.destroyAllWindows()
