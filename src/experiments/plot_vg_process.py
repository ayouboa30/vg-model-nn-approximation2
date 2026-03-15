import numpy as np

def plot_process(
    t: np.ndarray,
    x: np.ndarray,
):
    import matplotlib.pyplot as plt

    ax = plt.subplot()

    ax.plot(
        t, 
        x, 
        label="VG Process", 
        marker=".", 
        markerfacecolor="none",
        linestyle="none", 
        color="black"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
