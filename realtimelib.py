import joblib
import pywt
import pywt.data
import numpy as np
import math

mode = pywt.Modes.smooth

def find_accel(data):
    Akselerasi = []
    for i in data:
        aks = (i[2] ** 2 + i[3] ** 2 + i[4] ** 2) ** 0.5
        Akselerasi.append(aks)
    return Akselerasi

def extraction(data, w):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    mean_a = []
    maximum_a = []
    minimum_a = []
    stdev_a = []
    median_a = []
    var_a = []


    mean_coefA = []
    max_coefA = []
    min_coefA = []
    stdev_coefA = []
    median_coefA = []
    var_coefA = []


    mean_d = []
    maximum_d = []
    minimum_d = []
    stdev_d = []
    median_d = []
    var_d = []


    mean_coefD = []
    max_coefD = []
    min_coefD = []
    stdev_coefD = []
    median_coefD = []
    var_coefD = []



    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []


    idx = 1

    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)


        mean_a = np.mean([i[1] for i in ca])
        maximum_a = np.amax([i[1] for i in ca])
        minimum_a = np.amin([i[1] for i in ca])
        stdev_a = np.std([i[1] for i in ca])
        median_a = np.median([i[1] for i in ca])
        var_a = np.var([i[1] for i in ca])

        mean_coefA.append(mean_a)
        max_coefA.append(maximum_a)
        min_coefA.append(minimum_a)
        stdev_coefA.append(stdev_a)
        median_coefA.append(median_a)
        var_coefA.append(var_a)

        mean_d = np.mean([i[1] for i in cd])
        maximum_d = np.amax([i[1] for i in cd])
        minimum_d = np.amin([i[1] for i in cd])
        stdev_d = np.std([i[1] for i in cd])
        median_d = np.median([i[1] for i in cd])
        var_d = np.var([i[1] for i in cd])


        mean_coefD.append(mean_d)
        max_coefD.append(maximum_d)
        min_coefD.append(minimum_d)
        stdev_coefD.append(stdev_d)
        median_coefD.append(median_d)
        var_coefD.append(var_d)

        idx += 1

    feature_extraction = [min_coefD[4], max_coefD[4], mean_coefD[4], stdev_coefD[4], median_coefD[4], var_coefD[4]]
    return feature_extraction

def getAngleAcc(acX, acY, acZ):
    # TODO it seems that right now the acc data are not correctly
    # scaled so that's why I'm using gravity=0.98
    g = 0.980665
    pi = 3.141592
    # ATTENTION atan2(y,x) while in excel is atan2(x,y)
    r = math.atan2(acY, math.sqrt(acZ ** 2 + acX ** 2)) * 180 / pi
    p = math.atan2(acX, math.sqrt(acY ** 2 + acZ ** 2)) * 180 / pi
    # ATTENTION the following calculation does not return
    # a consistent value.
    # by the way it is not used
    y = math.atan2(math.sqrt(acX ** 2 + acY ** 2), acZ) * 180 / pi
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
        self.x = self.x + np.dot(K, y)
        # update I
        I = np.eye(self.n)
        # update P
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(
            np.dot(K, self.R), K.T)
        # return K, y, S, I

    # def getK(self):
    #     return self.K

def prepareData(data):
    acX = []
    acY = []
    acZ = []
    gyX = []
    gyY = []
    gyZ = []
    for row in data:
        acX.append(float(row[0]))
        acY.append(float(row[1]))
        acZ.append(float(row[2]))
        gyX.append(float(row[3]))
        gyY.append(float(row[4]))
        gyZ.append(float(row[5]))
    return acX, acY, acZ, gyX, gyY, gyZ

# Method to filter data using Kalman Filter class
def filterData(acX, acY, acZ):
    # Initializing parameters
    p_acX = []
    p_acY = []
    p_acZ = []
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
        p = kf.predict()
        p_acX.append(np.dot(H, p)[0])
        kf.update(z)
    # iterating through all element of list acY, then append filtered value to p_acY, then update Kalman Filter
    for z in acY:
        p_acY.append(np.dot(H, kf.predict())[0])
        kf.update(z)
    # # iterating through all element of list acZ, then append filtered value to p_acZ, then update Kalman Filter
    for z in acZ:
        p_acZ.append(np.dot(H, kf.predict())[0])
        kf.update(z)
    return p_acX, p_acY, p_acZ

def hitung_akselerasi(p_acX, p_acY, p_acZ):
    p_akselerasi = []
    for i in range(len(p_acX)):
        p_akselerasi.append(math.sqrt(p_acX[i] ** 2 + p_acY[i] ** 2 + p_acZ[i] ** 2))
    return p_akselerasi

def classify(data, waveletcoef="db2"):
    classifier = joblib.load('model.pkl')
    acX, acY, acZ, gyX, gyY, gyZ = prepareData(data)

    p_acX, p_acY, p_acZ = filterData(acX, acY, acZ)
    akselerasi_raw = hitung_akselerasi(acX, acY, acZ)
    akselerasi_filtered = hitung_akselerasi(p_acX, p_acY, p_acZ)
    feature_raw = extraction(akselerasi_raw, waveletcoef)
    feature_filtered = extraction(akselerasi_filtered, waveletcoef)
    predicted = classifier.predict([feature_raw, feature_filtered])
    output_raw = ""
    output_filtered = ""
    if predicted[0] == 2:
        output_raw = "Berjalan"
    if predicted[0] == 1:
        output_raw = "Berdiri"
    if predicted[0] == 3:
        output_raw = "Duduk"
    if predicted[1] == 2:
        output_filtered = "Berjalan"
    if predicted[1] == 1:
        output_filtered = "Berdiri"
    if predicted[1] == 3:
        output_filtered = "Duduk"
    return output_raw, output_filtered
