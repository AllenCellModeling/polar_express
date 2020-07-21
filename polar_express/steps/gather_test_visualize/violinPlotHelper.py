import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

# Helper methods for violin plot: finding IQR values
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

# Helper method to make violin plots. Creates 'num_compartments' violins corresponding to the various compaartments determined
# by the 'method' being used. Takes either 'AB' or 'Angular' as the method.
def makeViolinPlot(method, num_compartments, data, y_label, structure, num_samples, path):
    if (method != 'AB' and method != 'Angular'):
        raise Exception('Invalid mode entered. Please use \'AB\' or \'Angular\'.')

    # Plot the results
    fig, ax1 = plt.subplots()

    ax1.set_title(method + ' Compartments: ' + str(structure)
                    + '\n(' + str(num_compartments) + ' compartments, ' + str(num_samples) + ' samples)')
    ax1.set_ylabel(y_label)
    ax1 = sns.violinplot(data=data)

    labels = ['Sec ' + str(i) for i in range(num_compartments)]
    ax1.get_xaxis().set_tick_params(direction='out')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(np.arange(0, len(labels)))
    ax1.set_xticklabels(labels)

    cols = [data[:,i] for i in range(data.shape[1])]
    kruskal_h, kruskal_p = sp.stats.kruskal(*cols)

    ax1.text(0.95, 0.01, 'Kruskal-Wallis p-value: ' + str(kruskal_p), verticalalignment='bottom',
        horizontalalignment='right', transform=ax1.transAxes)

    plt.axhline(y=0, color='r', linestyle='-')

    # save visualizations
    if path.is_file():
        path.unlink()

    plt.savefig(path)
