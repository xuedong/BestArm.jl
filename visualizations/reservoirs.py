import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from labellines import labelLines
from scipy.stats import beta


# def stick_legend(axis=None):
#
#     if axis is None:
#         axis = plt.gca()
#
#     n = 32
#     nlines = len(axis.lines)
#     print(nlines)
#
#     xmin, xmax = axis.get_xlim()
#     ymin, ymax = axis.get_ylim()
#
#     # the 'point of presence' matrix
#     pop = np.zeros((nlines, n, n), dtype=np.float)
#
#     for l in range(nlines):
#         # get xy data and scale it to the NxN squares
#         xy = axis.lines[l].get_xydata()
#         xy = (xy - [xmin, ymin]) / ([xmax-xmin, ymax-ymin]) * n
#         xy = xy.astype(np.int32)
#         # mask stuff outside plot
#         mask = (xy[:, 0] >= 0) & (xy[:, 0] < n) & (xy[:, 1] >= 0) & (xy[:, 1] < n)
#         xy = xy[mask]
#         # add to pop
#         for p in xy:
#             pop[l][tuple(p)] = 1.0
#
#     # find whitespace, nice place for labels
#     ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0
#     # don't use the borders
#     ws[:, 0] = 0
#     ws[:, n-1] = 0
#     ws[0, :] = 0
#     ws[n-1, :] = 0
#
#     # blur the pop's
#     for l in range(nlines):
#         pop[l] = ndimage.gaussian_filter(pop[l], sigma=n/5)
#
#     for l in range(nlines):
#         # positive weights for current line, negative weight for others....
#         w = -0.3 * np.ones(nlines, dtype=np.float)
#         w[l] = 0.5
#
#         # calculate a field
#         p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
#         plt.figure()
#         plt.imshow(p, interpolation='nearest')
#         plt.title(axis.lines[l].get_label())
#
#         pos = np.argmax(p)  # note, argmax flattens the array first
#         best_x, best_y = pos / n, pos % n
#         x = xmin + (xmax-xmin) * best_x / n
#         y = ymin + (ymax-ymin) * best_y / n
#
#         axis.text(x, y, axis.lines[l].get_label(),
#                   horizontalalignment='center',
#                   verticalalignment='center')


def nu_0(x):
    """
    A wavy function as an example of reservoir.

    :param x: some real number
    :return: y
    """
    return 5*np.sin(x) + 2*x - x**2 + .3*x**3 - .2*(x-15)**4 - 10*x**2 * np.cos(x/3+12)**3 + .5*(x-12)**4


if __name__ == '__main__':
    x_values = np.linspace(10.0, 19.0, 100000)
    y_values = np.array([nu_0(x) for x in x_values])

    plt.close('all')

    sns.set()

    plt.plot(x_values, y_values, label=r'$\nu_0$')
    labelLines(plt.gca().get_lines())
    plt.xticks(np.arange(10, 20), ('0', '', '', '', '', '', '', '', '', r'$\mu^\star$'))
    plt.yticks([])
    plt.xlabel('means')
    plt.title('resevoir of arms')

    plt.savefig('../misc/figs/reservoir.pdf', bbox_inches='tight')

    # x_0 = np.linspace(beta.ppf(0.01, 5, 1), beta.ppf(0.99, 5, 1), 100)
    # x_1 = np.linspace(beta.ppf(0.01, 1, 4), beta.ppf(0.99, 1, 4), 100)
    # x_2 = np.linspace(beta.ppf(0.01, 1, 3), beta.ppf(0.99, 1, 3), 100)
    # x_3 = np.linspace(beta.ppf(0.01, 4, 5), beta.ppf(0.99, 4, 5), 100)
    # x_4 = np.linspace(beta.ppf(0.01, 2, 7), beta.ppf(0.99, 2, 7), 100)
    # plt.plot(x_0, beta.pdf(x_0, 5, 1), '--', label=r'$Beta(t-|\mathcal{L}_{t-1}|, 1)$')
    # plt.plot(x_1, beta.pdf(x_1, 1, 4))
    # plt.plot(x_2, beta.pdf(x_2, 1, 3))
    # plt.plot(x_3, beta.pdf(x_3, 4, 5))
    # plt.plot(x_4, beta.pdf(x_4, 2, 7))
    # plt.legend()
    #
    # plt.savefig('../misc/figs/order_trick.pdf')
