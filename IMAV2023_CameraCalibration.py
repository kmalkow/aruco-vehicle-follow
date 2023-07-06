# --------------------------------------------------------------------------
# Author:           Kevin Malkow
# Date:             05/07/23
# Affiliation:      TU Delft, IMAV 2023
#
# Version:          1.0 
# 
# Description:  
# Do a camera calibration and retrieve camera matrix and 
#  -------------------------------------------------------------------------

import glob
import cv2
import numpy as np

# Termination criteria for the calibration function
maxIter = 30
epsilon = 0.001
termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxIter, epsilon)


# -------------- Image Rescaling Function --------------
def img_rescale(img, scale_percent):
# Image downscaling -> USED AS IMAGE INPUT AND AFFECTS PERFORMANCE OF DETECTOR
  scale_percent = 30 # Percent of original size
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


#  ------------------------------------------------------------------------- #
#                           CONSTANTS DEFINITION                             #
#  ------------------------------------------------------------------------- #
# Number of intersection points of squares on long side of calibration board
widthCB = 9

# Number of intersection points of squares on short side of calibration board
heightCB = 6

# Size of square on chessboard
square_size = 0.022 # [m]

# # Initialise counter for formatting saved images -> UNCOMMENT TO SAVE IMAGES
# counter = 1

#  ------------------------------------------------------------------------- #
#                           CHESSBOARD CORNERS                               #
#  ------------------------------------------------------------------------- #
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((heightCB * widthCB, 3), np.float32)
objp[:, :2] = np.mgrid[0:widthCB, 0:heightCB].T.reshape(-1,2)

# Get actual positions of objects
objp = objp * square_size
 
# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

path = '/home/kevin/IMAV2023/Camera_Calibration/*.JPG'   
images = glob.glob(path)
 
for file in images:
  img = cv2.imread(file)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # Find the chess board corners
  ret, corners = cv2.findChessboardCorners(img_gray, (widthCB, heightCB), None)
  
  # If found, add object points, image points (after refining them)
  if ret == True:
    objpoints.append(objp)
  
    corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), termCriteria)
    imgpoints.append(corners2)
  
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (widthCB, heightCB), corners2, ret)
    
    #  ------------------------------------------------------------------------- #
    #                         DRAW AND SAVE IMAGES                               #
    #  ------------------------------------------------------------------------- #
    # # UNCOMMENT TO VIEW IMAGES
    # img = img_rescale(img, 30)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
  
    # # --------------------------------------------------------
    # # UNCOMMENT TO SAVE IMAGES
    # # Saving the image
    # cv2.imwrite('/home/kevin/IMAV2023/Camera_Calibration/Results/Pictures/ChessBoard_Detected_{}.JPG'.format(counter), img)
    # # --------------------------------------------------------
    
    # # UNCOMMENT TO VIEW IMAGES
    # cv2.destroyAllWindows()

  # # Increment counter -> UNCOMMENT TO SAVE IMAGES
  # counter += 1

#  ------------------------------------------------------------------------- #
#                           CAMERA CALIBRATION                               #
#  ------------------------------------------------------------------------- #
# Calibrate camera -> [..., camera matrix, distortion coefficients, rotation vectors, translation vectors]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

#  ------------------------------------------------------------------------- #
#                             SAVE VARIABLES                                 #
#  ------------------------------------------------------------------------- #
pathStore = '/home/kevin/IMAV2023/CameraCalibration_Variables/cameraCalibration.xml'   
cv_file = cv2.FileStorage(pathStore, cv2.FILE_STORAGE_WRITE)
cv_file.write("cM", mtx)
cv_file.write("dist", dist)

cv_file.release()

#  ------------------------------------------------------------------------- #
#                           ERROR COMPUTATION                                #
#  ------------------------------------------------------------------------- #
# Compute error using L2 Norm -> Error should be as close as possible to 0
totError = 0
for i in range(len(objpoints)):
  imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
  error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
  totError += error
 
print(f"Total Error: {totError/len(objpoints)}")











# #  ------------------------------------------------------------------------- #
# #                                FUNCTIONS                                   #
# #  ------------------------------------------------------------------------- #

# # -------------- Frame Rescaling Function --------------
# def frame_rescale(frame, scale_percent):
# # Frame downscaling -> USED AS FRAME INPUT AND AFFECTS PERFORMANCE OF DETECTOR
#   scale_percent = 30 # Percent of original size
#   width = int(frame.shape[1] * scale_percent / 100)
#   height = int(frame.shape[0] * scale_percent / 100)
#   dim = (width, height)
#   return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

# #  ------------------------------------------------------------------------- #
# #                            VIDEO PROCESSING                                #
# #  ------------------------------------------------------------------------- #

# # -------------- Load Video Live-stream --------------
# # Create a VideoCapture object and read from camera (input is either 0 or 1, for first and second camera, respectively)
# cap = cv2.VideoCapture(2)

# # -------------- Detect Aruco Markers, Write Video, and View Video --------------
# # Check if camera opened successfully
# if (cap.isOpened()== False): 
#   print("Error: cannot open video file or stream")
 
# # Default resolutions of the frame are obtained and converted from float to integer
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # Define video codec (FOURCC code)
# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

# # Create VideoWriter object 
# out = cv2.VideoWriter('/home/kevin/IMAV2023/Live_Video/ArucoMarker_LIVEVideo_Detected_1.mp4', fourcc, 30, (frame_width, frame_height))

# # Read until video is completed
# while(cap.isOpened()):
#   # Capture frame-by-frame
#   ret, frame = cap.read()
  
#   # Rescale frame
# #   frame = frame_rescale(frame, 30)

#   if ret == True:
#     # -------------- Aruco Marker Detection --------------
#     # Load dictionary for aruco marker
#     arucoDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

#     # Initiate aruco marker detection parameters
#     arucoParameters =  cv2.aruco.DetectorParameters()

#     # Aruco marker detection setup
#     arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, arucoParameters)

#     # List of aruco marker detected corners, IDs corresponding to each aruco marker, and rejected aruco markers
#     (markerCorners, markerIDs, rejectedCandidates) = arucoDetector.detectMarkers(frame)
    
#     # AT LEAST ONE MARKER DETECTED   
#     if len(markerCorners) > 0:
#       for i in range(0, len(markerIDs)):  
#         # Draw around the correctly detected aruco markers
#         cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIDs)  
            
#         # Draw around the rejected candidates
#         # cv2.aruco.drawDetectedMarkers(frame, rejectedCandidates, borderColor=(100, 200, 255))

#     # Write the frame into the file
#     out.write(frame)
    
#     # Display the resulting frame
#     cv2.imshow('Frame', frame)
 
#     # Wait 1 [ms] between each frame until it ends or press 'q' on keyboard to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
 
#   # Break the loop
#   else:
#     print('Error: frame not retrieved') 
#     break
 
# # Release the video capture and video write objects
# cap.release()
# out.release()
 
# # Close all the frames
# cv2.destroyAllWindows()