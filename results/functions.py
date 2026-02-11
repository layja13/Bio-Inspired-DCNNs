import pandas as pd
import matplotlib.pyplot as plt

# Function to plot CIFAR-10G results
def plot_cifar10g_results(data):
    # Get unique test sets and models
    test_sets = data['Set'].unique()
    models = data['Model'].unique()
    
    # Setup for subplots layout
    num_test_sets = len(test_sets)
    num_rows = 3
    num_cols = 2

    # Get a colormap with unique colors for each model
    colors = plt.cm.get_cmap('tab10', len(models)) 

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 10))
    
    # Iterate over each test set
    for i, test_set in enumerate(test_sets):
        # Determine subplot position (row, column)
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        
        # Filter data for the current test set
        subset = data[data['Set'] == test_set]
        
        # Create a list to store accuracies for each model
        accuracies = []
        for model in models:
            # Filter data for the current model
            model_subset = subset[subset['Model'] == model]
            if not model_subset.empty:
                # Add accuracy if data is available
                accuracies.append(model_subset['Accuracy'].values[0])
            else:
                # Assume 0 accuracy if no data is available
                accuracies.append(0)
        
        # Plot the bars with specific colors for each model
        ax.bar(models, accuracies, label=test_set, color=[colors(j) for j in range(len(models))])
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{test_set}')
        ax.grid(True)
    
    # Remove empty subplots if there are fewer than 6 test sets
    if num_test_sets < num_rows * num_cols:
        for j in range(num_test_sets, num_rows * num_cols):
            fig.delaxes(axes.flatten()[j])
    
    # Adjust space between subplots and save the figure
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig(f'CIFAR_10G.png')
    plt.show()


# Function to plot CIFAR-10 perturbed results by noise type in subplots
def plot_perturbed(data):
    noise_types = data['Noise'].unique()  # Get unique noise types
    num_noise_types = len(noise_types)
    
    # Determine the number of rows and columns based on the number of noise types
    num_cols = 2
    num_rows = (num_noise_types + num_cols - 1) // num_cols
    
    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easier access
    
    # Iterate over each noise type
    for i, noise_type in enumerate(noise_types):
        ax = axes[i]
        plot_cifar10_perturbed_results(data, noise_type, ax)
    
    # Remove empty subplots if there are more subplots than noise types
    if num_noise_types < len(axes):
        for j in range(num_noise_types, len(axes)):
            fig.delaxes(axes[j])

    test_set = data['Set'].unique()  # Get the name of the test set
    plt.tight_layout()
    plt.savefig(f'{test_set[0]}_perturbed.png')  # Save the figure with high resolution
    plt.show()


# Function to plot results for a specific noise type
def plot_cifar10_perturbed_results(data, noise_type, ax):
    # Filter data by the current noise type
    subset = data[data['Noise'] == noise_type]
    models = data['Model'].unique()  # Get unique models

    # Plot the results for each model
    for model in models:
        model_subset = subset[subset['Model'] == model]
        ax.plot(model_subset['Level'], model_subset['Accuracy'], marker='o', label=model)
    
    # Set specific labels for certain noise types
    if noise_type in ["Invert", "Rotation"]:
        ax.set_xlabel('Perturbation level')
    if noise_type in ["High Pass", "Contrast", "Darken", 'Rotation', 'Uniform']:
        ax.set_ylabel('Accuracy')

    ax.set_title(f'{noise_type}')  # Set the subplot title with the noise type
    ax.legend()  # Show legend
    ax.grid(True)  # Show grid
