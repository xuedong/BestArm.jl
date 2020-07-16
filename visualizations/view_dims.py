import matplotlib.pyplot as plt
import seaborn as sns

from labellines import labelLines


if __name__ == '__main__':
    sns.set()
    dims = [2, 3, 4, 5, 6]
    l_t3c = [3032.21, 3464.84, 3309.41, 3831.95, 3737.47]
    lingape = [9667.23, 12017.76, 15554.54, 17455.01, 19189.21]
    fig = plt.figure()
    plt.plot(dims, l_t3c, marker='o', label="L-T3C-Greedy")
    plt.plot(dims, lingape, marker='x', label="LinGapE")
    plt.legend()
    plt.ylabel('averaged stopping time')
    plt.xlabel('dimensions')
    plt.savefig("dims.pdf")
