import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


def makeViolinPlot(
    method, num_compartments, data, y_label, structure, num_samples, path
):

    """

    Helper method to make violin plots. Creates 'num_compartments' violins corresponding
    to the various compaartments determined by the 'method' being used. Takes either
    'AB' or 'Angular' as the method.

    Parameters
    ----------
    method : str
        "AB" if AB compartments is the method to visualize
        "Angular" if angular compartments is the method to visualize
    num_compartments : int
        the number of compartments (violins)
    data : numpy array where each column represents a section of the cell, and each
        row is a sample
    y_label : str
        label of the y-axis
    structure : str
        the structure that the GFP tags
    num_samples : int
        the number of images (samples) that the data contains
    path : str
        path to the directory that the visualization should be saves to

    """

    if method != "AB" and method != "Angular":
        raise Exception("Invalid mode entered. Please use 'AB' or 'Angular'.")

    fig, ax1 = plt.subplots()

    ax1.set_title(
        method
        + " Compartments: "
        + str(structure)
        + "\n("
        + str(num_compartments)
        + " compartments, "
        + str(num_samples)
        + " samples)"
    )
    ax1.set_ylabel(y_label)
    ax1 = sns.violinplot(data=data)

    labels = ["Sec " + str(i) for i in range(num_compartments)]
    ax1.get_xaxis().set_tick_params(direction="out")
    ax1.xaxis.set_ticks_position("bottom")
    ax1.set_xticks(np.arange(0, len(labels)))
    ax1.set_xticklabels(labels)

    cols = [data[:, i] for i in range(data.shape[1])]
    kruskal_h, kruskal_p = sp.stats.kruskal(*cols)

    ax1.text(
        0.95,
        0.01,
        "Kruskal-Wallis p-value: " + str(kruskal_p),
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax1.transAxes,
    )

    plt.axhline(y=0, color="r", linestyle="-")

    # save visualization
    if path.is_file():
        path.unlink()

    plt.savefig(path)
