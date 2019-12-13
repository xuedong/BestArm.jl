import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import visualizations.view_posterior as vp


def plot_vector(vectors):
    rows, cols = vectors.T.shape
    # print(rows, cols)

    # Get absolute maxes for axis ranges to center origin
    # This is optional
    maxes = 3 * np.amax(abs(vectors), axis=0)
    colors = ['b', 'r', 'k']
    fig = plt.figure(figsize=(7, 7))

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Pathological instance')

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    colors = ['b', 'r', 'k']

    for i, l in enumerate(range(0, cols)):
        # print(i)
        plt.axes().arrow(0, 0, vectors[i, 0], vectors[i, 1], head_width=0.05, head_length=0.1,
                         color=colors[i])

        ax.text(vectors[i, 0], vectors[i, 1], "x"+str(i+1),
                bbox={'facecolor': 'red', 'alpha': 0.2, 'pad': 0.5}, fontsize=8)

    plt.plot(0, 0, 'ok')  # <-- plot a black point at the origin
    # plt.axis('equal')  #<-- set the axes to the same scale
    plt.xlim([-maxes[0], maxes[0]])  # <-- set the x axis limits
    plt.ylim([-maxes[1], maxes[1]])  # <-- set the y axis limits

    plt.grid(b=True, which='major')  # <-- plot grid lines
    # plt.show()
    return fig, ax


if __name__ == "__main__":
    sns.set()
    x1 = np.array([1, 0])
    x2 = np.array([0, 1])
    x3 = np.array([np.cos(0.1), np.sin(0.1)])
    theta = np.array([2, 0.6])

    fig, ax_nstd = plot_vector(np.vstack((x1, x2, x3)))
    plt.axes().arrow(0, 0, theta[0], theta[1], head_width=0.02, head_length=0.02, linestyle='-.', color='k')
    ax_nstd.text(theta[0], theta[1], r'$\theta$')

    # fig, ax_nstd = plt.subplots(figsize=(6, 6))
    dependency_nstd = np.array([
        [3.001738321260551e-5, -0.0010014905698102542],
        [-0.0010014905698102542, 0.2000014203982155]
    ])
    mu = 2.0034260825126817, 0.19221341286927895
    scale = 3, 3

    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)

    x, y = vp.get_correlated_dataset(500, dependency_nstd, mu, scale)
    # ax_nstd.scatter(x, y, s=0.5)

    vp.confidence_ellipse(x, y, ax_nstd, n_std=1,
                          label=r'$1\sigma$', edgecolor='firebrick')
    vp.confidence_ellipse(x, y, ax_nstd, n_std=2,
                          label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    vp.confidence_ellipse(x, y, ax_nstd, n_std=3,
                          label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    plt.savefig("./instance.pdf")
