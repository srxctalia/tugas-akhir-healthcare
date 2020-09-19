import numpy as np
import csv
import math
import matplotlib.pyplot as plt

test =[]

acX = []
acY = []
acZ = []
gyX = []
gyY = []
gyZ = []
roll = []
pitch = []
yaw = []
time = []

p_acX = []
p_acY = []
p_acZ = []
p_gyX = []
p_gyY = []
p_gyZ = []
p_roll = []
p_pitch = []
p_yaw = []

k_ax = []
k_ay = []
k_az = []
k_gx = []
k_gy = []
k_gz = []
k_roll = []
k_pitch = []
k_yaw = []

p_akselerasi = []

global input_filename
global output_filename

def getAngleGyro(r,p,y,gyX,gyY,gyZ,dt):
    new_r=r+gyX*dt
    new_p=p+gyY*dt
    new_y=y+gyZ*dt
    return new_r,new_p,new_y

def getAngleAcc(acX, acY, acZ):
    # TODO it seems that right now the acc data are not correctly
    # scaled so that's why I'm using gravity=0.98
    g = 0.980665
    pi = 3.141592
    # ATTENTION atan2(y,x) while in excel is atan2(x,y)
    r = math.atan2(acY, math.sqrt(acZ**2 + acX**2)) * 180 / pi
    p = math.atan2(acX, math.sqrt(acY**2 + acZ**2 )) * 180 / pi
    # ATTENTION the following calculation does not return
    # a consistent value.
    # by the way it is not used
    y = math.atan2(math.sqrt(acX**2 + acY**2), acZ) * 180 / pi
    return r, p, y

def getAngleCompl(r, p, y, acX, acY, acZ, gyX, gyY, gyZ, dt):
    tau = 0.003
    # tau is the time constant in sec
    # for time periods <tau the  gyro takes precedence
    # for time periods > tau the acc takes precedence

    new_r, new_p, new_y = getAngleAcc(acX, acY, acZ)

    a = tau / (tau + dt)

    new_r = a * (new_r + gyX * dt) + (1 - a) * r
    new_p = a * (new_p + gyY * dt) + (1 - a) * p
    # note the yaw angle can be calculated only using the
    # gyro that's why a=1
    a = 1
    new_y = a * (new_y + gyZ * dt) + (1 - a) * y

    return new_r, new_p, new_y

# Kalman Filer purpose
class KalmanFilter(object):
    # Initializing Kalman Filter object
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        if (F is None or H is None):
            raise ValueError("Set proper system dynamics.")
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H

        # B is 0 if not set
        self.B = 0 if B is None else B
        # Q is identity matrix (sized equal to n) if not set
        self.Q = np.eye(self.n) if Q is None else Q
        # R is identity matrix (sized equal to n) if not set
        self.R = np.eye(self.n) if R is None else R
        # P is identity matrix (sized equal to n) if not set
        self.P = np.eye(self.n) if P is None else P
        # Q is zero matrix (sized equal to n) if not set
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    # Method to predict future value (as X)
    def predict(self, u=0):
        # predict x
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        # predict P
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # return x as predicted value
        return self.x

    # Method to update some Kalman Filter parameters
    def update(self, z):
        # upate y
        y = z - np.dot(self.H,
                       self.x)
        # update S
        S = self.R + np.dot(self.H,
                            np.dot(self.P,
                                   self.H.T))
        # update K
        K = np.dot(np.dot(self.P,
                          self.H.T),
                   np.linalg.inv(S))

        # print('Kalman Gain :',K)
        # print(len(K))
        # update x
        self.x = self.x + np.dot(K,y)
        # update I
        I = np.eye(self.n)
        # update P
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return K , y, S, I
    def getK(self):
        return self.K


def prepareData():
    # acX	acY	acZ	gyX	gyY	gyZ	time
    with open(input_filename + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                acX.append(float(row[0]))
                acY.append(float(row[1]))
                acZ.append(float(row[2]))
                gyX.append(float(row[3]))
                gyY.append(float(row[4]))
                gyZ.append(float(row[5]))
                time.append(row[6])
            line_count += 1
    csv_file.close()

def writeData():
    # No     waktu   accx    accY    accZ    gyroX   gyroY   gyroZ   roll    pitch
    with open(output_filename + '.csv', mode='w', newline='') as output_file:
        w = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['No', 'Waktu', 'accx', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ','akselerasi', 'roll', 'pitch', 'yaw'])

        for i in range(len(time)):
            w.writerow([i + 1,
                        time[i],
                        float(p_acX[i]),
                        float(p_acY[i]),
                        float(p_acZ[i]),
                        float(p_gyX[i]),
                        float(p_gyY[i]),
                        float(p_gyZ[i]),
                        float(p_akselerasi[i]),
                        float(p_roll[i]),
                        float(p_pitch[i]),
                        float(p_yaw[i])])
    output_file.close()

    with open(output_filename + '_kg.csv', mode='w', newline='') as output_file:
        w = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['No', 'kg_acx1', 'kg_acx2', 'kg_acx3', 'kg_acy1', 'kg_acy2', 'kg_acy3', 'kg_acz1','kg_acz2', 'kg_acz3'])

        for i in range(len(time)):
            w.writerow([i + 1,
                        float(k_ax[i][0]),
                        float(k_ax[i][1]),
                        float(k_ax[i][2]),
                        float(k_ay[i][0]),
                        float(k_ay[i][1]),
                        float(k_ay[i][2]),
                        float(k_az[i][0]),
                        float(k_az[i][1]),
                        float(k_az[i][2])])
    output_file.close()

def fusionSensor():
    dt = 1.0 / 60
    for i in range(len(time)):
        if i == 0:
            j = getAngleCompl(0, 0, 0, acX[i], acY[i], acZ[i], gyX[i], gyY[i], gyZ[i], dt)
            pitch.append(j[1])
            yaw.append(j[2])
            roll.append(j[0])
        else:
            j = getAngleCompl(
                roll[i - 1],
                pitch[i - 1],
                yaw[i - 1],
                acX[i],
                acY[i],
                acZ[i],
                gyX[i],
                gyY[i],
                gyZ[i],
                dt
            )

            pitch.append(j[1])
            yaw.append(j[2])
            roll.append(j[0])

# Method to filter data using Kalman Filter class
def filterData():
    # Initializing parameters
    # initialize delta t
    dt = 1.0 / 60
    # initialize F matrix (3x3)
    F = np.array([[1, dt, 0],
                  [0, 1, dt],
                  [0, 0, 1]])
    # initialize H matrix (1x3)
    H = np.array([1, 0, 0]).reshape(1, 3)
    # initialize Q matrix (3x3)
    Q = np.array([[0.05, 0.05, 0.0],
                  [0.05, 0.05, 0.0],
                  [0.0, 0.0, 0.0]])
    # initialize R matrix (1x1)
    R = np.array([0.5]).reshape(1, 1)

    # Kalman Filter object
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)

    # iterating through all element of list acX, then append filtered value to p_acX, then update Kalman Filter
    for z in acX:
        p_acX.append(np.dot(H, kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        # print(Q)
        k_ax.append(m[0])
        # k_ax.append(test[z])
    # iterating through all element of list acY, then append filtered value to p_acY, then update Kalman Filter
    for z in acY:
        p_acY.append(np.dot(H, kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_ay.append(m[0])
    # # iterating through all element of list acZ, then append filtered value to p_acZ, then update Kalman Filter
    for z in acZ:
        p_acZ.append(np.dot(H, kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_az.append(m[0])
    # # iterating through all element of list gyX, then append filtered value to p_gyX, then update Kalman Filter
    for z in gyX:
        p_gyX.append(np.dot(H, kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_gx.append(m[0])
    # iterating through all element of list gyY, then append filtered value to p_gyY, then update Kalman Filter
    for z in gyY:
        p_gyY.append(np.dot(H, kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_gy.append(m[0])
    # iterating through all element of list gyZ, then append filtered value to p_gyZ, then update Kalman Filter
    for z in gyZ:
        p_gyZ.append(np.dot(H, kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_gz.append(m[0])
    # iterating through all element of list roll, then append filtered value to p_roll, then update Kalman Filter
    for z in roll:
        p_roll.append(np.dot(H,  kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_roll.append(m[0])
    # iterating through all element of list pitch, then append filtered value to p_pitch, then update Kalman Filter
    for z in pitch:
        p_pitch.append(np.dot(H,  kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_pitch.append(m[0])
    # iterating through all element of list yaw, then append filtered value to p_yaw, then update Kalman Filter
    for z in yaw:
        p_yaw.append(np.dot(H,  kf.predict())[0])
        kf.update(z)
        m = kf.update(z)
        k_yaw.append(m[0])


def plotData():
    #plt.plot(range(len(acX)), np.array(acX), label='accX')
    #plt.plot(range(len(acY)), np.array(acY), label='accY')
    #plt.plot(range(len(acZ)), np.array(acZ), label='accZ')
    #plt.plot(range(len(gyX)), np.array(gyX), label='gyroX')
    #plt.plot(range(len(gyY)), np.array(gyY), label='gyroY')
    plt.plot(range(len(yaw)), np.array(yaw), label='yaw')
    plt.plot(range(len(roll)), np.array(roll), label = 'roll')
    plt.plot(range(len(pitch)), np.array(pitch), label = 'pitch')

    #plt.plot(range(len(p_acX)), np.array(p_acX), label='p_accX')
    #plt.plot(range(len(p_acY)), np.array(p_acY), label='p_accY')
    #plt.plot(range(len(p_acZ)), np.array(p_acZ), label='p_accZ')
    #plt.plot(range(len(p_gyX)), np.array(p_gyX), label='p_gyroX')
    #plt.plot(range(len(p_gyY)), np.array(p_gyY), label='p_gyroY')
    plt.plot(range(len(p_yaw)), np.array(p_yaw), label='p_yaw')
    plt.plot(range(len(p_roll)), np.array(p_roll), label = 'p_roll')
    plt.plot(range(len(p_pitch)), np.array(p_pitch), label = 'p_pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (Degress)')
    plt.legend()
    plt.show()

def hitung_akselerasi():
    for i in range(len(time)):
        p_akselerasi.append(math.sqrt(p_acX[i]**2+p_acY[i]**2+p_acZ[i]**2))

# def calculateSNR(signal, signalNNoise):
#     snr = 0
#     noise = []
#     for i in range(len(signal)):
#         noise.append(abs(signal[i] - signalNNoise[i]))
#     for i in range(len(signal)):
#         snr += signal[i]/noise[i]
#     return snr / len(signal)

if __name__ == '__main__':
    input_filename = input('Masukkan nama file input: ')
    output_filename = input('Masukkan nama file output: ')
    prepareData()
    fusionSensor()
    filterData()
    plotData()
    hitung_akselerasi()
    writeData()

    # print('Kalman Gain acX :',k_ax)
    # MSE = np.square(np.subtract(pitch, p_pitch)).mean()
    # SNR = calculateSNR(p_pitch, pitch)
    # print('MSE:  ', MSE)
    # print('SNR:  ', SNR)
    print(k_ax)
    
