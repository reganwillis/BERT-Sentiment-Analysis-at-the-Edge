import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv('fork-of-bertforsentimentanalysis.log', delimiter=' ')
#time linenum type arrow epoch epochnum step stepnum losslabel loss acclabel acc
log = log.drop(columns=['time', 'linenum', 'arrow', 'epoch', 'epochnum', 'step', 'stepnum', 'losslabel', 'acclabel'])
print(log)

type_list = log['type'].tolist()
loss = log['loss'].tolist()
acc = log['acc'].tolist()

print(type_list[0])
print(loss[0])
print(acc[0])

# remove comma at the end of loss column
for i in range(len(loss)):
    loss[i] = loss[i].split(',')[0]
print(loss[0])

# sort into appropriate lists
train_loss = []
val_loss = []
train_acc = []
val_acc = []
for i in range(len(type_list)):
    if type_list[i] == "TRAINING":
        train_loss.append(loss[i])
        train_acc.append(acc[i])
    elif type_list[i] == "VALIDATION":
        val_loss.append(loss[i])
        val_acc.append(acc[i])
    else:
        print("ERROR: type not recognized:", type_list[i])

for i in range(5):
    print(train_loss[i])
    print(val_loss[i])
    print(train_acc[i])
    print(val_acc[i])
    print("\n")

print(len(train_loss), len(val_loss), len(train_acc), len(val_acc))

def plot_epoch_metrics(x, y, data_names, title_prefix, yaxis_label):
    """Plot metrics with the number of epochs on the x axis and the metric of
    interest on the y axis. Note that this function differs based on the input.

    :param x: The values to use on the x-axis.
    :type x: list
    :param y: A list of lists containing len(x) data points to plot. The inner
        lists are the different series to plot.
    :type y: list
    :param data_names: Names of the series to use in the legend.
    :type data_names: str
    :param title_prefix: A prefix to add before everything else in the title.
    :type title_prefix: str
    :param yaxis_label: The label for the y axis.
    :type yaxis_label: str
    """
    # Plot multiple series of data
    for i in y:
        plt.plot(x, i)
    # Set the title
    plt.title(title_prefix + ' ' + ' vs. '.join(data_names) + ' ' + yaxis_label)
    # Set the y axis label
    plt.ylabel(yaxis_label)
    # Enable the legend with the appropriate names
    plt.legend(data_names)

NUM_EPOCHS = 4
plot_epoch_metrics(
    np.arange(NUM_EPOCHS),
    [train_loss, val_loss],
    ['Train', 'Validation'],
    'BERT - Finetuning for Sentiment Analysis',
    'Loss'
)

plot_epoch_metrics(
    np.arange(NUM_EPOCHS),
    [train_acc, val_acc],
    ['Train', 'Validation'],
    'BERT - Finetuning for Sentiment Analysis',
    'Accuracy'
)
