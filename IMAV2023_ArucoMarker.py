#  This file is part of paparazzi
# 
#  paparazzi is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2, or (at your option)
#  any later version.
# 
#  paparazzi is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with paparazzi; see the file COPYING.  If not, see
#  <http://www.gnu.org/licenses/>.
# 

# -------------------------------------------------------------
#  @file "modules/aruco_marker/IMAV2023_ArucoMarker.py"
#  Detect Aruco marker and track it (pose estimation of camera)
#  -------------------------------------------------------------
import numpy as np
import cv2
import cv2.aruco

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
