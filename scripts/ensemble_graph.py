import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

#sns.set_theme(style="whitegrid")

labels = ['Base', 'Ensemble', 'Points Based', 'Confidence Based']
data = [69, 72, 76, 80]
#ax = sns.barplot(x="Method", y="Accuracy", data=data,order=labels)
#sns.show()

#exit()

x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, data, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracies')
ax.set_title('Accuracy by Method')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)

fig.tight_layout()

plt.show()