import numpy as np
from matplotlib import pyplot as plt


accuracies = np.load('../accuracies.npy')
print(accuracies)
exit()
j1 = [i[0] for i in accuracies]
j2 = [i[1] for i in accuracies]
j3 = [i[2] for i in accuracies]
j4 = [i[3] for i in accuracies]
j5 = [i[4] for i in accuracies]
j6 = [i[5] for i in accuracies]
j7 = [i[6] for i in accuracies]
j8 = [i[7] for i in accuracies]
j9 = [i[8] for i in accuracies]
j_all = [i[-2] for i in accuracies]
j_all_method2 = [i[-1] for i in accuracies]
j1_smoothed = []
j2_smoothed = []
j3_smoothed = []
j4_smoothed = []
j5_smoothed = []
j6_smoothed = []
j7_smoothed = []
jall_smoothed = []
jall_method2_smoothed = []
j1s = []
j2s = []
j3s = []
j4s = []
j5s = []
j6s = []
j7s = []
j8s = []
j9s = []
jalls = []
jalls_method2 = []
for i in range(100):
    j1s.append(j1[i])
    j2s.append(j2[i])
    j3s.append(j3[i])
    j4s.append(j4[i])
    j5s.append(j5[i])
    j6s.append(j6[i])
    j7s.append(j7[i])
    j8s.append(j8[i])
    j9s.append(j9[i])
    jalls.append(j_all[i])
    jalls_method2.append(j_all_method2[i])
    if i%10 == 0:
        j1_smoothed.append(np.average(j1s))
        j2_smoothed.append(np.average(j2s))
        j3_smoothed.append(np.average(j3s))
        j4_smoothed.append(np.average(j4s))
        j5_smoothed.append(np.average(j5s))
        j6_smoothed.append(np.average(j6s))
        j7_smoothed.append(np.average(j7s))
        jall_smoothed.append(np.average(jalls))
        jall_method2_smoothed.append(np.average(jalls_method2))
        j1s = []
        j2s = []
        j3s = []
        j4s = []
        j5s = []
        j6s = []
        j7s = []
        j8s = []
        j8s = []
        jalls = []
        jalls_method2 = []
all_smoothed = [np.average([j1_smoothed[i], j2_smoothed[i], j3_smoothed[i], j4_smoothed[i], j5_smoothed[i], j6_smoothed[i], j7_smoothed[i]]) for i in range(len(j1_smoothed))]
plt.plot(j1_smoothed, label="Individual Judge Accuracy Average")
plt.plot(jall_smoothed, label="Expert Guess Method 1")
plt.plot(jall_method2_smoothed, label="Expert Guess Method 2")
plt.legend()
plt.ylabel('Accuracy')
plt.title('MNIST 4 digit Classification')
plt.xlabel('10 epoch accuracy')
x = 5

j1avg = np.average(j1[x:])
j2avg = np.average(j2[x:])
j3avg = np.average(j3[x:])
j4avg = np.average(j4[x:])
j5avg = np.average(j5[x:])
j6avg = np.average(j6[x:])
j7avg = np.average(j7[x:])
j8avg = np.average(j8[x:])
j9avg = np.average(j9[x:])
jallavg = np.average(j_all[x:])
jm2avg = np.average(j_all_method2[x:])
print(np.average([j1avg, j2avg, j3avg, j4avg, j5avg, j6avg, j7avg, j8avg, j9avg]))
print(jallavg)
print(jm2avg)
print(np.max(j_all_method2))
plt.show()