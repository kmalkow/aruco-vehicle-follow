import pymap3d
import numpy as np
import sys
import time 
import math

sys.path.append("/home/sergio/paparazzi/sw/ext/pprzlink/lib/v2.0/python/")

from ivy.std_api import *
import pprzlink.ivy
import pprzlink.messages_xml_map as messages_xml_map
import pprzlink.message as message       


pitch_values = None         # Global variable to store Ivybus received pitch values
roll_values = None          # Global variable to store Ivybus received pitch values
yaw_values = None           # Global variable to store Ivybus received pitch values


# --------- Bind to Drone Attitude Message --------- # 
def attitude_callback(ac_id, pprzMsg):
    global pitch_values
    global roll_values
    global yaw_values
    
    pitch = pprzMsg['theta']
    roll  = pprzMsg['phi']
    yaw   = pprzMsg['psi']
    pitch_values = pitch
    roll_values  = roll
    yaw_values   = yaw

# --------- Get Attitude Values --------- # 
def get_attitude_values():
    global pitch_values
    global roll_values
    global yaw_values
    return pitch_values, roll_values, yaw_values


                                      # Ivybus INITIALISATION #
# ------------------------------------------------------------------------------------------------------- #
# --------- Create Ivy Interface --------- # 
ivy = pprzlink.ivy.IvyMessagesInterface(agent_name="ArucoMarker", start_ivy=False, ivy_bus="127.255.255.255:2010")

# --------- Start Ivy Interface --------- # 
ivy.start()

# --------- Subscribe to Ivy Messages --------- # 
ivy.subscribe(attitude_callback, message.PprzMessage("telemetry", "NPS_RATE_ATTITUDE"))

                                      # Time Delay Before Loop #
# ------------------------------------------------------------------------------------------------------- #


pitch, roll, yaw = get_attitude_values()

while pitch is None and roll is None and yaw is None:
    pitch, roll, yaw = get_attitude_values()

while True:
    pitch, roll, yaw = get_attitude_values()

    pitch_fl = float(pitch)
    roll_fl = float(roll)
    yaw_fl = float(yaw)

    pitch_rad = math.radians(pitch_fl)
    roll_rad = math.radians(roll_fl)
    yaw_rad = math.radians(yaw_fl)

    print("pitch: ", pitch_rad)
    print("roll: ", roll_rad)
    print("yaw: ", yaw_rad)
    print("")

    
    # Drone Outputs
    lat_drone = 1  # Replace with actual latitude of the drone
    lon_drone = 1  # Replace with actual longitude of the drone
    alt_drone = 1  # Replace with actual altitude of the drone

    # Local coordinates of the ArUco marker with respect to the drone
    local_x = 2  # Replace with actual local x-coordinate
    local_y = 3  # Replace with actual local y-coordinate
    local_z = 4  # Replace with actual local z-coordinate

    local_vector = np.array([[local_x], [local_y], [local_z]])

    # STEP 2: Define Rotation Matrix (Yaw, Pitch, Roll)
    def ned_vector_calc(yaw, pitch, roll, local_vector):
        # Yaw (rotation around z-axis)
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Pitch (rotation around y-axis)
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Roll (rotation around x-axis)
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_combined = R_yaw @ R_pitch @ R_roll 

        ned_vector = np.dot(R_combined, local_vector)
        north, east, down = ned_vector.squeeze()

        return north, east, down

    # Convert local coordinates to NED coordinates with respect to the drone
    ned_north, ned_east, ned_down = ned_vector_calc(yaw_rad, pitch_rad, roll_rad, local_vector)

    # Print NED coordinates
    print("NED Coordinates (North, East, Down):", ned_north, ned_east, ned_down)


ivy.shutdown()

