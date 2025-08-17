import matplotlib.pyplot as plt

def line_plot(x, name=""):
    plt.plot(range(len(x)), x)
    plt.savefig(f"plots/{name}")
    plt.xlabel("Rewards")
    plt.ylabel("Steps")
    