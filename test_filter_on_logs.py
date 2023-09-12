from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

from filterV1 import init, predict, correct

# READ LOGFILES

f_N = open("Measured_Variables/Outdoor_Tests/IMAV_09_09_23_TEST1_ArucoNORTH_CompleteV2")
f_E = open("Measured_Variables/Outdoor_Tests/IMAV_09_09_23_TEST1_ArucoEAST_CompleteV2")

N = np.loadtxt(f_N, delimiter=",", dtype=str).astype(float)
E = np.loadtxt(f_E, delimiter=",", dtype=str).astype(float)


# SIMULATED MAIN LOOP AT 15Hz

tM = N[:,1]
print("Simulation end time",tM[-1])
dt = 1.0 / 15.0
tsim = np.arange(0,tM[-1],dt)



# LOGGING FOR PLOT
pf_N = []
pf_E = []
pf_VN = []
pf_VE = []


# init
x = init( [N[0,0], E[0,0], 25] );

i = 0
for t in np.nditer(tsim):
    # predict
    x = predict(dt)
    #print('predict x=', x)
    
    if tM[i] < t:
        # correct
        Z = [N[i,0], E[i,0], 25]
        #print('Z', Z)
        x = correct( Z )
        #print('correct x=', x)

        while tM[i] < t:
            i+=1
        

    pf_N.append(float(x[0]))
    pf_E.append(float(x[1]))
    pf_VN.append(float(x[2]))
    pf_VE.append(float(x[3]))



plt.plot(N[:,0], E[:,0],'x')
plt.plot(pf_N, pf_E)
#plt.plot(pf_N, pf_E, '*')
plt.ylabel('some numbers')
plt.grid()
plt.show()


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(tsim, pf_N)
axs[0, 0].set_title('N')
axs[0, 0].grid()
axs[0, 1].plot(tsim, pf_E, 'tab:orange')
axs[0, 1].set_title('E')
axs[0, 1].grid()
axs[1, 0].plot(tsim, pf_VN, 'tab:green')
axs[1, 0].set_title('VN')
axs[1, 0].grid()
axs[1, 1].plot(tsim, pf_VE, 'tab:red')
axs[1, 1].set_title('VE')
axs[1, 1].grid()
plt.show()
