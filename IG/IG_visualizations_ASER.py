import numpy as np
import matplotlib.pyplot as plt

def visualize_integrated_gradients(attributions, feature_names=None, aggregation, file_name):
    """
    Visualize the Integrated Gradients attributions.
    Parameters:
        attributions: Torch tensor or NumPy array containing attributions.
        feature_names: List of feature names (optional).
        aggregation: Aggregation method if attributions are multidimensional ('mean', 'sum', 'max').
    """
    # Convert to NumPy array if not already
    if hasattr(attributions, "detach"):
        attributions = attributions.detach().cpu().numpy()

    # Debug: Print the shape before aggregation
    #print("Shape before aggregation:", attributions.shape)

    # Handle multidimensional attributions
    if attributions.ndim > 1:
        if aggregation == 'mean':
            attributions = attributions.mean(axis=1)  # Aggregate over the second dimension (axis 1)
        elif aggregation == 'sum':
            attributions = attributions.sum(axis=1)
        elif aggregation == 'max':
            attributions = attributions.max(axis=1)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

    # Debug: Print the shape after aggregation
    #print("Shape after aggregation:", attributions.shape)

    # Squeeze the attributions to ensure it's one-dimensional
    attributions = attributions.squeeze()  # Remove extra dimensions

    # Ensure the attributions are now one-dimensional
    if attributions.ndim != 1:
        raise ValueError("Attributions must be one-dimensional after aggregation.")

    # Create indices for features
    feature_indices = np.arange(len(attributions))

    plt.rcParams.update({
    'font.size': 30,            # General font size
    'axes.titlesize': 30,       # Title font size
    'axes.labelsize': 30,       # Label font size
    'xtick.labelsize': 35,      # X-axis tick font size
    'ytick.labelsize': 35       # Y-axis tick font size
    })

    # Plot the attributions
    plt.figure(figsize=(30, 10))  # Increase the width of the figure
    plt.bar(feature_indices, attributions, color='skyblue', edgecolor='black', width = 1)
    plt.xlabel('Feature Index' if feature_names is None else 'Feature Name')
    plt.ylabel('Attribution')
    plt.title('Integrated Gradients')

    # Add feature names if provided
    '''if feature_names:
        if len(feature_names) != len(attributions):
            raise ValueError("Number of feature names must match the number of attributions.")
        plt.xticks(feature_indices, feature_names, rotation=45, ha="right")'''

    if feature_names:
        if len(feature_names) != len(attributions):
            raise ValueError("Number of feature names must match the number of attributions.")
        # Show only every nth label for clarity
        n = 5  # Adjust this value as needed
        plt.xticks(feature_indices[::n], feature_names[::n], rotation=45, ha="right")
    else:
        # If no feature names, adjust spacing of feature indices
        n =5  # Adjust this value as needed
        plt.xticks(feature_indices[::n], feature_indices[::n], rotation=45, ha="right")

    

    plt.tight_layout()  # Adjust layout
    #plt.show()

    plt.savefig(file_name)  # Save the plot
    plt.close()  # Close the plot to avoid display if not needed

    print(f"Plot saved as {file_name}")