from matplotlib import pyplot as plt
plt.style.use('ggplot')
import numpy as np

num_classes =18

if num_classes ==4:
    labels = ['OneVOne', 'Ensemble', 'Solution']
    mnist = [63,72,85]
    fashion = [42.46,69,77.6]
    cifar = [37.113,37.5,48]


    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    Base = [37.113, 42.46, 63]
    Ensemble = [37.5, 69, 72]
    Solution = [48, 77.6, 85]

    # Set position of bar on X axis
    br1 = np.arange(len(Base))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, Base, color='crimson', width=barWidth,
            edgecolor='grey', label='OneVOne')
    plt.bar(br2, Ensemble, color='mediumblue', width=barWidth,
            edgecolor='grey', label='Ensemble')
    plt.bar(br3, Solution, color='darkorange', width=barWidth,
            edgecolor='grey', label='Solution')

    # Adding Xticks
    plt.xlabel('Dataset', fontweight='bold', fontsize=15)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(Base))],
               ['Cifar10', 'Fashion', 'MNIST'])
    plt.title('4 Digit Classification Accuracy by Method')
    plt.legend()
    plt.show()

if num_classes == 8:
    labels = ['OneVOne', 'Ensemble', 'Solution']

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    OneVOne = [21.5, 22, 28]
    Ensemble = [26.5, 30.3, 33.1]
    Solution = [35.3, 60, 73.5]

    # Set position of bar on X axis
    br1 = np.arange(len(OneVOne))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, OneVOne, color='crimson', width=barWidth,
            edgecolor='grey', label='OneVOne')
    plt.bar(br2, Ensemble, color='mediumblue', width=barWidth,
            edgecolor='grey', label='Ensemble')
    plt.bar(br3, Solution, color='darkorange', width=barWidth,
            edgecolor='grey', label='Solution')

    # Adding Xticks
    plt.xlabel('Dataset', fontweight='bold', fontsize=15)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(OneVOne))],
               ['Cifar10', 'Fashion', 'MNIST'])
    plt.title('8 Digit Classification Accuracy by Method')
    plt.legend()
    plt.show()
if num_classes == 2:
    labels = ['One-V-One', 'Ensemble', 'Solution']

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    OneVOne = [76.8, 94, 98]
    Ensemble = [80.4, 94.2, 99.6]
    Solution = [80.4, 94.2, 99.6]
    # Set position of bar on X axis
    br1 = np.arange(len(OneVOne))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    # Make the plot
    plt.bar(br1, OneVOne, color='crimson', width=barWidth,
            edgecolor='grey', label='OneVOne')
    plt.bar(br2, Ensemble, color='mediumblue', width=barWidth,
            edgecolor='grey', label='Ensemble')
    plt.bar(br3, Solution, color='darkorange', width=barWidth,
            edgecolor='grey', label='Solution')

    # Adding Xticks
    plt.xlabel('Dataset', fontweight='bold', fontsize=15)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(OneVOne))],
               ['Cifar10', 'Fashion', 'MNIST'])
    plt.title('2 Digit Classification Accuracy by Method')
    plt.legend()
    plt.show()
if num_classes ==10:
    Members = [15, 10, 3, 1]
    #NN = [89, 85, 65, 61]
    NN = [1, .955, .7303, .685]
    #Q = [36.5, 36.5, 35, 35.9]
    Q = [1, .967, .846, .83]
    plt.plot(Members, NN, color='blue', marker='o', label="Classical Ensemble")
    plt.plot(Members, Q, color='red', marker='o', label="Quantum Ensemble")
    plt.gca().invert_xaxis()
    plt.title('Ensemble Accuracy by Members', fontsize=14)
    plt.xlabel('Members', fontsize=14)
    plt.ylabel('Normalized Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()


if num_classes == 11:
    classes = [2,3,4,5,6,7,8, 9, 10]
    #Accs = [74.07, 63.701, 47.933, 39.351, 33.797, 27.833]
    Models_Needed = [1, 3, 6, 10, 15, 21, 28, 36, 45]
    #plt.plot(classes, Accs, color='green', marker='o', label="Accuracy")
    plt.plot(classes, Models_Needed, color='blue', marker='o', label="Models Needed")
    plt.title('One Vs One Classification Scalability', fontsize=14)
    plt.xlabel('Number of Classes', fontsize=14)
    plt.ylabel('Models Needed', fontsize=14)
    plt.grid(True)
    plt.show()


if num_classes == 12:
    labels = ['Single Member 4', 'Ensemble 4']

    barWidth = 0.3
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    solo4 = [30.8, 28.8, 47.9]
    ens4 = [30, 32, 58.5]

    # Set position of bar on X axis
    br1 = np.arange(len(solo4))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, solo4, color='red', width=barWidth,
            edgecolor='grey', label='4 Classes Single Member')
    plt.bar(br2, ens4, color='blue', width=barWidth,
            edgecolor='grey', label='4 Classes Solution')

    # Adding Xticks
    plt.xlabel('Dataset', fontweight='bold', fontsize=15)
    plt.ylabel('Inference Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + (barWidth/2) for r in range(len(solo4))],
               ['Lima', 'Manila', 'Interpolated Prediction'])
    plt.title('Real Quantum Machine Classification Accuracy')
    plt.legend()
    plt.show()

if num_classes == 12.5:
    solo8 = [17.2, 18.4, 24.7]
    ens8 = [22, 26, 47]
    onevone = [22, 26, 47]
    labels = ['Single Member 8', 'Ensemble 8', 'OneVOne']

    barWidth = 0.3
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(solo8))
    br2 = [x + barWidth for x in br1]


    # Make the plot
    plt.bar(br1, solo8, color='red', width=barWidth,
            edgecolor='grey', label='8 Classes Single Member')
    plt.bar(br2, ens8, color='blue', width=barWidth,
            edgecolor='grey', label='8 Classes Solution')

    # Adding Xticks
    plt.xlabel('Dataset', fontweight='bold', fontsize=15)
    plt.ylabel('Inference Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + (barWidth/2) for r in range(len(solo8))],
               ['Lima', 'Manila', 'Interpolated Prediction'])
    plt.title('Real Quantum Machine Classification Accuracy')
    plt.legend()
    plt.show()

if num_classes == 13:
    classes = [2,3,4,5,6,7,8]
    Accs = [99.6, 74.07, 63.701, 47.933, 39.351, 33.797, 27.833]
    Models_Needed = [1, 3, 6, 10, 15, 21, 28, 36, 45]
    plt.plot(classes, Accs, color='green', marker='o', label="Accuracy")
    plt.title('One Vs One Classification Accuracy', fontsize=14)
    plt.xlabel('Number of Classes', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True)
    plt.show()

if num_classes == 14:
    plt.style.use('ggplot')
    Members = range(1,16)
    Same = [46.5, 51, 52, 57.5, 58.5, 61.5, 58.5, 61, 59, 59.5, 61, 62.5, 62, 62, 61.5]
    Mixed =[46.6, 58.6, 62.6, 63.6, 64.4, 62, 62, 64.8, 64.8, 64, 65, 65.8, 65.2, 66.2, 66.6]

    plt.plot(Members, Same, color='blue', marker='o', label="Same Architecture")
    plt.plot(Members, Mixed, color='red', marker='o', label="Heterogeneous Ensembles")
    plt.title('Ensemble Accuracy by Members', fontsize=14)
    plt.xlabel('Members', fontsize=14)
    plt.ylabel('8-Class MNIST Classification Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

if num_classes == 15:
    plt.style.use('ggplot')
    Methods = ['One-v-One', 'Base', 'Ensemble',  'Multi-Bit', 'Mixed Arc', 'Full Solution']
    x_pos = [i for i, _ in enumerate(Methods)]
    accs = [28, 32.6, 33.1, 65.6, 66.6, 72]
    plt.bar(x_pos, accs, color='green')
    plt.xlabel('Methods')
    plt.ylabel('Accuracy')
    plt.title('MNIST 8 Digit Classification by Method')
    plt.xticks(x_pos, Methods)
    plt.show()
if num_classes == 16:
    labels = ['OneVOne', 'Ensemble', 'Solution']

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    class_2 = [98, 99.6, 99.6]
    class_4 = [63, 72, 85]
    class_8 = [28, 33.5, 73.5]


    OneVOne = [98, 63, 28]
    Ensemble = [99.6, 72, 33.5]
    Solution = [99.6, 85, 73.5]

    # Set position of bar on X axis
    br1 = np.arange(len(class_2))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, OneVOne, color='crimson', width=barWidth,
            edgecolor='grey', label='OneVOne')
    plt.bar(br2, Ensemble, color='mediumblue', width=barWidth,
            edgecolor='grey', label='Ensemble')
    plt.bar(br3, Solution, color='darkorange', width=barWidth,
            edgecolor='grey', label='Solution')

    # Adding Xticks
    plt.xlabel('Dataset', fontweight='bold', fontsize=15)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(class_2))],
               ['2 Class', '4 Class', '8 Class'])
    plt.title('MNIST Classification Accuracy')
    plt.legend()
    plt.show()
if num_classes == 17:
    x = [0, .005, .01, .015, .02, .025, .03, .035, .04, .045, .05, .06, .07]
    y = [74, 72, 68.5, 66.5, 65, 63, 60, 57, 56, 56, 52, 43, 34.5]
    plt.plot(x,y)
    plt.title("Simulated Error Classification for MNIST - 8")
    plt.xlabel("Error Rate")
    plt.ylabel("Accuracy")
    plt.scatter(x=.07, y=34.5, c='g', label='Real Machine Acc')
    plt.scatter(x=0, y=74, c='b', label='No Noise - Ideal Case')
    plt.legend()
    plt.show()

if num_classes == 18:
    x = [0, .01, .015, .02, .03]
    y = [74, 67, 54, 47, 41]
    plt.plot(x,y)
    plt.title("Simulated Error Classification for MNIST - 8 Bitflips")
    plt.xlabel("Error Rate")
    plt.ylabel("Accuracy")
    plt.scatter(x=0, y=74, c='b', label='No Noise - Ideal Case')
    plt.legend()
    plt.show()