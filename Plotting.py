import matplotlib
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
#from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["solution_cost", "expanded", "generated", "weights"])
BaselineStats = namedtuple("BLStats",["solution_cost", "expanded", "generated"])

def plot_stats(stats, blstates, rstats, smoothing_window=30, noshow=False):

    # Plot the solution cost over time
    fig1 = plt.figure(figsize=(10,5))
    cost_smoothed = pd.Series(stats.solution_cost).rolling(smoothing_window, min_periods=smoothing_window).mean()
    sol_cost_plt = plt.plot(cost_smoothed, color='b', linestyle='-', label="Learnt Heurisitic")
    #bl_smoothed = pd.Series(blstates.solution_cost).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #sol_exp_plt = plt.plot(bl_smoothed, color='r', linestyle='--', label="Greedy")
    weights_smoothed = pd.Series(stats.weights * 100).rolling(smoothing_window, min_periods = smoothing_window).mean()
    plt.plot(weights_smoothed, color='g', linestyle='-', label="weights")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Solution cost (Smoothed)")
    plt.title("Solution Cost over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the expanded count over time
    fig2 = plt.figure(figsize=(10,5))
    expanded_smoothed = pd.Series(stats.expanded).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(expanded_smoothed, color='b', linestyle='-', label="Learnt Heurisitic")
    #bl_smoothed = pd.Series(blstates.expanded).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #plt.plot(bl_smoothed, color='r', linestyle='--', label="Greedy")
    weights_smoothed = pd.Series(stats.weights * 1000).rolling(smoothing_window, min_periods = smoothing_window).mean()
    plt.plot(weights_smoothed, color='g', linestyle='-', label="weights")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Expanded count (Smoothed)")
    plt.title("Expanded count over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the generated count over time
    fig3 = plt.figure(figsize=(10,5))
    generated_smoothed = pd.Series(stats.generated).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(generated_smoothed, color='b', linestyle='-')
    #bl_smoothed = pd.Series(blstates.generated).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #plt.plot(bl_smoothed, color='r', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Generated count (Smoothed)")
    plt.title("Generated count over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the solution cost over time
    fig4 = plt.figure(figsize=(10, 5))
    rcost_smoothed = pd.Series(rstats.solution_cost).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rcost_smoothed, color='b', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Solution cost ratio (Smoothed)")
    plt.title("Solution Cost ratio over Time  (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the expanded count over time
    fig5 = plt.figure(figsize=(10, 5))
    rexpanded_smoothed = pd.Series(rstats.expanded).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rexpanded_smoothed, color='b', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Expanded count ratio (Smoothed)")
    plt.title("Expanded count ratio over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the generated count over time
    fig6 = plt.figure(figsize=(10, 5))
    rgenerated_smoothed = pd.Series(rstats.generated).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rgenerated_smoothed, color='b', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Generated count ratio (Smoothed)")
    plt.title("Generated count ratio over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    return fig1, fig2, fig3, fig4, fig5, fig6
