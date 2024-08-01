import os

import numpy as np
from matplotlib import pyplot as plt


def plot_loss_acc(loss, val_loss, results_directory):
    # Find the index (epoch) of the lowest validation loss and the highest validation accuracy
    lowest_val_loss_epoch = np.argmin(val_loss)
    # highest_val_acc_epoch = np.argmax(val_acc)

    # Plot the metrics
    plt.plot(loss, label="loss")
    # plt.plot(acc, label='accuracy')
    plt.plot(val_loss, label="val_loss")
    # plt.plot(val_acc, label='val_accuracy')

    # Add markers for the lowest validation loss and the highest validation accuracy
    plt.plot(
        lowest_val_loss_epoch,
        val_loss[lowest_val_loss_epoch],
        marker="o",
        markersize=8,
        label="Lowest val_loss: "
        + str(round(val_loss[lowest_val_loss_epoch], 3))
        + ", ep "
        + str(lowest_val_loss_epoch),
        linestyle="None",
        color="red",
    )
    # plt.plot(highest_val_acc_epoch, val_acc[highest_val_acc_epoch], marker='o', markersize=8, label="Highest val_accuracy: " + str(round(val_acc[highest_val_acc_epoch], 3)) + ", ep " + str(highest_val_acc_epoch), linestyle='None', color='green')

    # Annotate the points with the epoch numbers
    plt.annotate(
        f"Epoch {lowest_val_loss_epoch}",
        (lowest_val_loss_epoch, val_loss[lowest_val_loss_epoch]),
        textcoords="offset points",
        xytext=(-10, 7),
        ha="center",
        fontsize=8,
        color="red",
    )
    # plt.annotate(f"Epoch {highest_val_acc_epoch}", (highest_val_acc_epoch, val_acc[highest_val_acc_epoch]), textcoords="offset points", xytext=(-10,-15), ha='center', fontsize=8, color='green')

    # Add the legend and save the plot
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.suptitle("Word2Sprite", fontsize=16)
    plt.savefig(os.path.join(results_directory, "loss_graph.png"))
    plt.close()
