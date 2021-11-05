from matplotlib import pyplot as plt
import numpy as np

all_data = np.load('../master_data.npy')
split_data = [i.split(',') for i in all_data[1:]]
weighted = [float(i[-1]) for i in split_data if i[1]=='weighted']
unweighted = [float(i[-1]) for i in split_data if i[1]=='unweighted']
partial = [float(i[-1]) for i in split_data if i[1]=='unweighted-partial']

single_member_weighted = []
ensemble_weighted = []
p_weighted = []
for i in range(250):
    if i%5==0:
        single_member_weighted.append(weighted[i])
    if i%5==4:
        ensemble_weighted.append(weighted[i])

for i in range(250):
    if i%5==0:
        pass
    if i%5==4:
        p_weighted.append(partial[i])

print(single_member_weighted[9])
print(single_member_weighted[49])

print(ensemble_weighted[9])
print(ensemble_weighted[49])
plt.plot(p_weighted, label='p_weighted')
plt.plot(ensemble_weighted, label='enweighted')
#plt.plot(partial, label='partial')

#plt.plot(single_member_weighted, label='singe member weighted')
#plt.plot(ensemble_weighted, label='ensemble weighted')
plt.legend()
plt.show()