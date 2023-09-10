# --------------------------------------------------------------------------
# Author:           Kevin Malkow and Sergio Marin Petersen
# Date:             08/09/23
# Affiliation:      TU Delft, IMAV 2023
#
# Version:          1.1 
# 
# Description:  
# - Detect Aruco markers and estimate Aruco marker position from real-time video stream
# - Get attitude and NED position from drone through Ivybus
# - Convert Aruco marker position in image plane coordinates to NED coordinates for navigation algorithm
#   using drone attitude
#
#  -------------------------------------------------------------------------

                                        # LIBRARY DEFINITION #
# ------------------------------------------------------------------------------------------------------- #
# --------- General --------- # 
import time
import math
import csv
import sys
import cv2
import cv2.aruco
import threading
import numpy as np
from typing import Tuple


                                  # FUNCTIONS -> NED COORDINATES CONVERSION #
# ------------------------------------------------------------------------------------------------------- #
def NED_conversion(pitch, roll, yaw, Aruco_position):
    # --------- Rotation Matrix (Rotation around Z-Axis/Yaw) ------- # 
    RX= np.array([
                 [np.cos(yaw), -np.sin(yaw), 0],
                 [np.sin(yaw), np.cos(yaw),  0],
                 [          0,           0,  1]])

    # --------- Rotation Matrix (Rotation around Y-Axis/Pitch) ------- # 
    RY = np.array([
                  [ np.cos(pitch),  0, np.sin(pitch)],
                  [             0,  1,             0],
                  [-np.sin(pitch),  0, np.cos(pitch)]])

    # --------- Rotation Matrix (Rotation around X-Axis/Roll) ------- # 
    RZ = np.array([
                  [1,            0,             0],
                  [0, np.cos(roll), -np.sin(roll)],
                  [0, np.sin(roll),  np.cos(roll)]])

    # --------- Rotation Matrix ------- # 
    R = RZ @ RY @ RX 

    # --------- Obtain NED Coordinates ------- # 
    NED_vector = np.dot(R, Aruco_position)
    NORTH, EAST, DOWN = NED_vector.squeeze()

    return NORTH, EAST, DOWN



# set a random pitch, drone, yaw of drone to test
# ------------------------------------------------------------------- #

PITCH_DRONE = 0
ROLL_DRONE = 0
YAW_DRONE = 0

# ------------------------------------------------------------------- #


# Set a random aruco marker position in camera coordinates
# ------------------------------------------------------------------- #

X_ARUCO = 3
Y_ARUCO = 3
Z_ARUCO = 3

# ------------------------------------------------------------------- #

X_ARUCO_B = Y_ARUCO
Y_ARUCO_B = -X_ARUCO
Z_ARUCO_B = Z_ARUCO

ARUCO_POSITION_B = np.array([[X_ARUCO_B], [Y_ARUCO_B], [Z_ARUCO_B]])

# --------- Convert Aruco Position in Image Coordinates to NED Coordinates Relative to Drone --------- # 
NORTH_ARUCO, EAST_ARUCO, DOWN_ARUCO = NED_conversion(PITCH_DRONE, ROLL_DRONE, YAW_DRONE, ARUCO_POSITION_B)

         
# NORTH EAST DOWN OF THE DRONE (choose as well)
# ------------------------------------------------------------------- #

NORTH_DRONE = 3
EAST_DRONE  = 3
DOWN_DRONE  = 3

# ------------------------------------------------------------------- #


# Addition of North East Down of Drone with Relative North East Down of Aruco

NORTH_COORDINATE = NORTH_ARUCO + NORTH_DRONE
EAST_COORDINATE = EAST_ARUCO + EAST_DRONE
DOWN_COORDINATE = DOWN_ARUCO + DOWN_DRONE


