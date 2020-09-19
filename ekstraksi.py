import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pywt
import pywt.data
import pandas as pd
import os

namedfile=[]
feature_extraction = []
all_feature = []

for root, dirs, files in os.walk("data_duduk/"):
    for filename in files:
        namedfile.append(filename)

for j in range(len(namedfile)):

    dataset = pd.read_csv("data_duduk/" + str(namedfile[j]))
    print ("read file: " + str(namedfile[j]))
    # ecg = pywt.data.ecg()

    data1 = dataset['Akselerasi']
    mode = pywt.Modes.smooth

    def plot_signal_decomp(data, w, title):
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

        for i in range(6):
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

        feature_extraction = [
            float("{:.3f}".format(min_coefD[4])),
            float("{:.3f}".format(max_coefD[4])),
            float("{:.3f}".format(mean_coefD[4])),
            float("{:.3f}".format(stdev_coefD[4])),
            float("{:.3f}".format(median_coefD[4])),
            float("{:.3f}".format(var_coefD[4])),
            '3'
        ]

        all_feature.append(feature_extraction)
        print("completed")

        rec_a = []
        rec_d = []


        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w))


        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w))


        fig = plt.figure()
        ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
        ax_main.set_title(title)
        ax_main.plot(data)
        ax_main.set_xlim(0, len(data) - 1)

    plot_signal_decomp(data1, 'db2', "DWT: Signal irregularity")
     #                    "DWT: Frequency and phase change - Daubechies 2")
    # plot_signal_decomp(data1, 'db2', "DWT: Ecg sample - Daubechies 2")




    # plt.show()

print(all_feature)
np.savetxt("data_train/train_data_duduk2.csv", all_feature, fmt='%s',header="min,max,mean,stdev,median,variance,label", delimiter=",")
