from recordtype import recordtype
import numpy as np
import matplotlib.pyplot as plt

# Define a mutable namedtuple for tasks
Task = recordtype('Task', 'weight duration due done')

# Create todo tasks with exponential due dates and random weights/durations
def create_tasks(num): 
    tasks = []
    distribution = np.round(np.random.exponential(5, 10000))
    for _ in range(num):
        weight = np.random.randint(1, 100)  # Priority weight
        due = np.random.choice(distribution) + 1  # Due date
        duration = np.random.randint(1, due)  # Duration < Due Date
        tasks.append(Task(weight=weight, duration=duration, due=due, done=0))
    return tasks

# Tally the results of the simulation
def tally(completed_tasks, all_tasks):
    completed = len(completed_tasks) / len(all_tasks) if all_tasks else 0

    # Identify important tasks (75th percentile based on weight)
    percentile = round(0.75 * len(all_tasks))
    important_tasks = sorted(all_tasks, key=lambda x: x.weight)[percentile:]

    # Calculate important tasks completed
    important = sum(1 for t in important_tasks if t in completed_tasks) / len(important_tasks) if important_tasks else 0

    # Calculate tasks completed in time
    intime = sum(1 for t in completed_tasks if t.done <= t.due) / len(all_tasks) if all_tasks else 0

    return (completed, important, intime)

# Single simulation run
def simulate(ratio, *tasks):
    # Reset the `done` status of tasks
    for task in tasks:
        task.done = 0

    deadline = ratio * sum(task.duration for task in tasks)
    completed_tasks = []
    elapsed_time = 0

    for task in tasks:
        elapsed_time += task.duration
        if elapsed_time <= deadline:
            task.done = elapsed_time
            completed_tasks.append(task)
        else:
            break

    return tally(completed_tasks, tasks)

# Run multiple simulations with a given algorithm
def run(ratio, iterations, num, algorithm):
    final_results = (0, 0, 0)

    for _ in range(iterations):
        tasks = create_tasks(num)
        results = simulate(ratio, *algorithm(*tasks))
        final_results = tuple(map(sum, zip(final_results, results)))

    return tuple(x / iterations for x in final_results)

# Display results of the simulation
def print_results(data):
    params = ("- Tasks completed", "- Important tasks completed", "- Tasks completed in time")
    for value, label in zip(data, params):
        print(f"{label.ljust(40)}: {value * 100:6.2f}%")

# Algorithms for task prioritization
def as_they_come(*tasks): return tasks
def due_first(*tasks): return sorted(tasks, key=lambda x: x.due)
def due_last(*tasks): return sorted(tasks, key=lambda x: x.due, reverse=True)
def important_first(*tasks): return sorted(tasks, key=lambda x: x.weight, reverse=True)
def easier_first(*tasks): return sorted(tasks, key=lambda x: x.duration)
def easier_important_first(*tasks): return sorted(tasks, key=lambda x: x.duration / x.weight)
def easier_due_first(*tasks): return sorted(tasks, key=lambda x: x.duration / x.due)

# Create radar charts to visualize results
def chart(data, titles):
    labels = np.array(['Completed Tasks', 'Important Tasks', 'In Time Tasks'])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    colors = ['blue', 'red', 'green', 'violet', 'orange', 'deepskyblue', 'darkgreen']

    for result, title, color in zip(data, titles, colors):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        result = np.concatenate((result, [result[0]]))  # Close the plot
        ax.plot(angles, result, 'o-', linewidth=2, color=color, label=title)
        ax.fill(angles, result, alpha=0.25, color=color)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(title, size=14, weight='bold', va='top')
        plt.show()

# Run the Monte Carlo simulation with configurable inputs
def run_simulation(ratio=0.5, iterations=100, num=25):
    algorithms = [
        as_they_come, due_first, due_last, important_first,
        easier_first, easier_important_first, easier_due_first
    ]
    labels = [
        "As They Come", "Due Tasks First", "Due Tasks Last",
        "Important Tasks First", "Easier Tasks First",
        "Easier Important Tasks First", "Easier Due Tasks First"
    ]

    data = []
    for algo, label in zip(algorithms, labels):
        print(label)
        results = run(ratio, iterations, num, algo)
        print_results(results)
        data.append(results)
        print()

    chart(data, labels)

# Example usage: Run the simulation with custom parameters
if __name__ == "__main__":
    # Customize these parameters as needed
    run_simulation(ratio=0.6, iterations=200, num=30)
