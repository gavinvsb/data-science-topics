import numpy as np
import pandas as pd
import math  
import matplotlib.pyplot as plt
from scipy import stats

# Tasks dictionary defined outside the function for easy customization
tasks = {
    "Task 1": [4, 7, 12],
    "Task 2": [3, 5, 8],  # Example: Add more tasks if needed
}

def simulate_task_completion(tasks, no_of_experiments=10_000, num_bins=100):
    """
    Simulate the completion time of tasks using beta distributions and plot the histogram.
    
    Parameters:
        tasks (dict): Dictionary with task names as keys and [optimistic, likely, pessimistic] estimates as values.
        no_of_experiments (int): Number of simulations to perform.
        num_bins (int): Number of bins for the histogram.
    """
    # Prepare the DataFrame from the task dictionary
    data = pd.DataFrame(tasks, index=["optimistic", "likely", "pessimistic"])
    
    # Initialize an array to store cumulative points from all tasks
    all_points = np.zeros(no_of_experiments)

    # Loop through each task and accumulate the completion times
    for task_name, (a, b, c) in tasks.items():
        # Calculate alpha and beta parameters for the beta distribution
        alpha = ((4 * b) + c - (5 * a)) / (c - a)
        beta = ((5 * c) - a - (4 * b)) / (c - a)

        # Generate random samples for the current task
        r = np.random.beta(alpha, beta, no_of_experiments)
        points = (r * (c - a)) + a  # Scale to the task's optimistic-pessimistic range
        all_points += points  # Accumulate the points from each task

    # Plot the histogram
    plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(
        all_points, 
        num_bins, 
        range=(0, np.max(all_points)), 
        color="skyblue", 
        lw=1, 
        edgecolor="steelblue", 
        weights=[100 / no_of_experiments] * no_of_experiments  # Normalize to sum to 100%
    )
    plt.xlabel("Completion Time")
    plt.ylabel("Probability Density (%)")
    plt.title("Task Completion Time Distribution")
    plt.show()

    # Analyze the statistics of the simulated data
    analyze_statistics(all_points)

    # Print probabilities for various standard deviations
    print("Probabilities")
    print("-------------")
    print_probability(bins, n, np.mean(all_points), np.std(all_points), "0.5σ", 0.5)
    print_probability(bins, n, np.mean(all_points), np.std(all_points), "1σ", 1)
    print_probability(bins, n, np.mean(all_points), np.std(all_points), "2σ", 2)
    print_probability(bins, n, np.mean(all_points), np.std(all_points), "3σ", 3)


def analyze_statistics(points):
    """
    Analyze and print statistical data of the simulated completion times.

    Parameters:
        points (ndarray): Array of simulated task completion times.
    """
    _, minmax, mean, var, skewness, kurtosis = stats.describe(points)
    sd = math.sqrt(var)

    print("\nStatistics")
    print("----------")
    print(f"minimum: {minmax[0]:.1f}, maximum: {minmax[1]:.1f}")
    print(f"mean: {mean:.1f}")
    print(f"standard deviation: {sd:.1f}")
    print(f"skewness: {skewness:.2f}")
    print(f"kurtosis: {kurtosis:.2f}")
    print()


def print_probability(bins, n, mean, sd, label, fraction):
    """
    Calculate and print the probability of the data falling within a specified range of standard deviations.

    Parameters:
        bins (ndarray): Array of bin edges from the histogram.
        n (ndarray): Array of counts in each bin.
        mean (float): Mean of the simulated data.
        sd (float): Standard deviation of the simulated data.
        label (str): Label for the standard deviation range.
        fraction (float): Fraction of the standard deviation.
    """
    upper = np.searchsorted(bins, mean + sd * fraction)
    lower = np.searchsorted(bins, mean - sd * fraction)
    prob = np.sum(n[lower:upper])  # Sum the weights in the range
    print(f"* between {mean - sd * fraction:.1f} and {mean + sd * fraction:.1f}: {prob:.1f}% ({label})")


# Run the simulation with user-defined parameters
simulate_task_completion(tasks, no_of_experiments=1_000_000, num_bins=100)
