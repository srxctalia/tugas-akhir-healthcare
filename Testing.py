import pandas as pd
import joblib
import pywt
import pywt.data
import numpy as np
import time
import matplotlib.pyplot as plt

start= time.process_time()

dataset = pd.read_csv("Data Pengujian/Data berdiri/Subjek_1.csv") #input Data
dataset = np.array(dataset)
Akselerasi = []
for i in dataset:
    aks = (i[2]**2 + i[3]**2 + i[4]**2)**0.5
    Akselerasi.append(aks)

mode = pywt.Modes.smooth
feature_extraction = []
all_feature = []

def plot_signal_decomp(data, w):
        global all_feature
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
        all_feature.append(feature_extraction)


classifier = joblib.load('model.pkl')
plot_signal_decomp(Akselerasi,"db2")
predicted = classifier.predict(all_feature)
print( 'Hasil Ekstraksi Ciri :',all_feature)
print('Deteksi :',predicted)
if predicted[0] == 2:
    print("Berjalan")
if predicted[0] == 1:
    print("Berdiri")
if predicted[0] == 3:
    print("Duduk")



end= time.process_time()
runtime = end - start
print('Runtime: '+ str(runtime) + 'detik')
plt.show()
