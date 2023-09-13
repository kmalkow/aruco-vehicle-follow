from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np


#############################################################################
## Smooth track

racetrack_at_competition = np.asarray([-31.9311054713794,
271.783338767474,
-32.0712612077143,
354.330707476033,
6.54081246301252,
383.589318217342,
69.7688247447596,
371.79730057929,
82.4293454201474,
317.952569707543,
41.2149464895494,
283.464634824714,
-19.0602618491586,
254.205805061623,
-55.9858755301192,
184.452443674776,
-112.958411922747,
93.2285411270681,
-216.217248725545,
-67.4128541508385,
-207.285803430584,
-102.679390260187,
-178.800825545386,
-157.081574962641,
-176.129043959345,
-189.121590913013,
-195.051149417886,
-212.928357567232,
-230.080445408295,
-219.156908501874,
-249.845234939409,
-206.584775444945,
-258.565968964039,
-179.661912656902,
-232.887278638881,
-68.7471303050234,
-129.135408800651,
97.678909647592,
-69.3491451054297,
198.915079108026,
-31.9311054713794,
271.783338767474])

test_field_track = np.asarray([-81.9,
133.7,
-43.4,
80.7,
27.7,
116.7,
-28.7,
183.9,
-141.7,
199.9,
-81.9,
133.7])

y = racetrack_at_competition
# y = test_field_track

n = len(y)
y = y.reshape((int(n/2),2))
n = len(y)
x = range(0, n)

tck1 = interpolate.splrep(x, y[:,0], s=0.001, k=3)
tck2 = interpolate.splrep(x, y[:,1], s=0.001, k=3)

# 10000 = 4m/s
x_new = np.linspace(min(x), max(x), 10000)
y_fitx = interpolate.BSpline(*tck1)(x_new)
y_fity = interpolate.BSpline(*tck2)(x_new)

fit = np.vstack((y_fitx, y_fity)).T

def find_closest_point( P ):
    global fit

    distances = np.sqrt((fit[:, 0] - P[0])**2 + (fit[:, 1] - P[1])**2)
    closest_index = np.argmin(distances)
    return closest_index


start = np.asarray([[-113.6], [67]])
nr = find_closest_point(start)
print('Start at:', nr)

def route():
    global nr
    global y_fitx
    global y_fity

    # Hack: move along the track blindly
    nr += 1
    if nr >= len(y_fitx):
        nr = 0

    zk = [y_fity[nr], y_fitx[nr], 0, 0 ]

    return zk



#############################################################################
## Kalman filter stuff

x = np.asarray([start[0],
                start[1],
                0,
                0])


H = np.asarray([[1,  0, 0 ,0],
                [0,  1, 0 ,0]] )


# print('A',A)
# print('H',H)

K0 = 1e5
Kp = 1
Kv = 0.0001
Kpv = 0
Km = 1e5

Q = np.asarray([[Kp, 0, Kpv, 0],
                [0, Kp, 0, Kpv],
                [Kpv, 0, Kv, 0],
                [0, Kpv, 0 ,Kv]])
R = np.asarray([[Km],[Km]])
P = np.asarray([[K0, 0, 0, 0],
                [0, K0, 0, 0],
                [0, 0,  0, 0],
                [0, 0,  0, 0]])


#print('P', P)
#print('Q', Q)
#print('R', R)

KP = 0.4
KV = 0.05



def init( X0 ):
    global x

    N = X0[0]
    E = X0[1]
    D = X0[2]

    x = np.asarray([[N],
                [E],
                [0],
                [0]])
    
    print('X0 set to:', x)



vision_update_counter = 0

def predict(dt):
    global x
    global P
    global Q
    global vision_update_counter
    global nr

    # Do Kalman predict
    A = np.asarray([[1,  0,  dt,  0],
                    [0,  1,  0,   dt],
                    [0,  0,  1,   0],
                    [0,  0,  0,   1]] )

    x = A @ x
    P = ((A @ P) @ A.T) + Q

    # Keep track of the number of predictions since the last update
    vision_update_counter += 1

    # On timeout, predict that the car follows the track
    TIMEOUT_SEC = 5
    FPS = 15
    if vision_update_counter == (TIMEOUT_SEC * FPS):
        # For every ArUco: update the track to the closest point
        nr = find_closest_point(x)

    if vision_update_counter >= (TIMEOUT_SEC * FPS):
        # Get the next position on the route as the next measurement
        z = route()

        # Update the filter as if we were following the track
        update(z, True)


    return x



def update(Z, route=False):
    global x
    global H
    global R
    global P
    global KP
    global KV
    global vision_update_counter

    if not route:
        vision_update_counter = 0

    zk = np.asarray([[Z[0]],
                     [Z[1]]])
    #print('z',zk)
    yk = zk - H@x
    S = H @ P @ H.T + R
    #print('S',S)
    #Si = np.linalg.inv(S)
    #print('S-1',Si)
    #K = (P @ H.T) @ Si
    K = np.asarray([[KP, 0],[0, KP],[KV, 0],[0, KV]])
    #print('K',K)
    x = x + (K @ yk)
    #print('K*yk',K @ yk)
    P = (np.eye(4) - (K @ H)) @ P
    #print('P',P)

    return x




