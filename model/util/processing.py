import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import altair as alt
import pandas as pd
def random_coreset(dataset, coreset_size):
    """
    Randomly selects a subset of data points to form a coreset.

    Args:
    - dataset (torch.utils.data.Dataset): The dataset to sample from.
    - coreset_size (int): The number of samples to include in the coreset.

    Returns:
    - coreset_indices (torch.Tensor): Indices of the selected samples.
    """
    # Ensure coreset size does not exceed dataset size
    coreset_size = min(coreset_size, len(dataset))
    
    # Randomly select indices without replacement
    coreset_indices = np.random.choice(len(dataset), size=coreset_size, replace=False)
    
    # Convert numpy array to torch tensor
    coreset_indices = torch.from_numpy(coreset_indices)
    
    coreset = Subset(dataset, coreset_indices)
    return coreset

def create_split_task(dataset, classes):
    """
    Create a binary classification task from the MNIST dataset.
    
    Parameters:
    - dataset: The original MNIST dataset (training or test).
    - classes: A tuple of two integers representing the classes to include in the split.
    
    Returns:
    - A Subset of the original dataset containing only the specified classes.
    """
    # Find indices of classes we're interested in
    indices = [i for i, (_, target) in enumerate(dataset) if target in classes]
    
    # Create a subset of the dataset with only the specified classes
    subset = Subset(dataset, indices)
    
    return subset

def create_split_dataloaders(train_dataset, test_dataset, tasks, batch_size=256):
    """
    Create DataLoaders for each binary task in Split MNIST.
    
    Parameters:
    - train_dataset: The MNIST training dataset.
    - test_dataset: The MNIST test dataset.
    - batch_size: The batch size for the DataLoader.
    
    Returns:
    - A list of tuples containing (train_loader, test_loader) for each binary task.
    """
    train_loaders = []
    test_loaders = []
    for task in tasks:
        # Create training subset and DataLoader
        train_subset = create_split_task(train_dataset, task)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # Create test subset and DataLoader
        test_subset = create_split_task(test_dataset, task)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


def permute_mnist(mnist, perm):
    """Apply a fixed permutation to the pixels of each image in the dataset."""
    permuted_data = []
    for img, target in mnist:
        # Flatten the image, apply permutation and reshape back to 1x28x28
        img_permuted = img.view(-1)[perm].view(1, 28, 28)
        permuted_data.append((img_permuted, target))
    return permuted_data

def plot_trends_with_autovcl(trends, betas,title='Permuated MNIST Experiment', lower=0.7):
    import pandas as pd
    import numpy as np
    import altair as alt

    # Insert a None at the beginning of the AutoBeta list to align with the second point onwards
    adjusted_auto_beta = [None] + list(np.log(np.mean(betas, axis=0)))

    df = pd.DataFrame({
        '# of tasks': range(1,len(trends[0][0])+1),
        'beta = 0.01': np.mean(trends[0], axis=0),
        'beta = 1': np.mean(trends[1], axis=0),
        'beta = 100': np.mean(trends[2], axis=0),
        'AutoVCL': np.mean(trends[3], axis=0),
        'AutoBeta': adjusted_auto_beta  # Use the adjusted AutoBeta data
    })
    axis_start = 1.5
    # Convert the DataFrame to long format for accuracies
    df_long_acc = df.melt('# of tasks', var_name='Series', value_name='Values', value_vars=['beta = 0.01', 'beta = 1', 'beta = 100', 'AutoVCL'])
    legend_order = ['beta = 0.01', 'beta = 1', 'beta = 100', 'AutoVCL']
    # Plot for accuracies
    acc_chart = alt.Chart(df_long_acc).mark_line(point=alt.OverlayMarkDef(size=100),
    strokeWidth=4  # Adjust line thickness here
    ).encode(
        x=alt.X('# of tasks:Q', title='# tasks', 
            scale=alt.Scale(domain=[axis_start, len(trends[0][0])]),  # Adjust domain slightly for visual alignment
            axis=alt.Axis(values=list(range(1, len(trends[0][0])+1)))
           ),
        y=alt.Y('Values:Q', scale=alt.Scale(domain=[lower, 1]), axis=alt.Axis(grid=True), title='Accuracy'),
        color=alt.Color('Series:N', sort=legend_order,scale=alt.Scale(scheme='category10'), legend=alt.Legend(title="Model")),
        tooltip=['# of tasks', 'Values', 'Series']
    )
    # Plot for AutoBeta
    loss_chart = alt.Chart(df).mark_bar(opacity=0.3, color='skyblue', width=40).encode(
        x=alt.X('# of tasks:Q', title='# tasks', 
            scale=alt.Scale(domain=[axis_start, len(trends[0][0])]),  # Adjust domain slightly for visual alignment
            axis=alt.Axis(values=list(range(1, len(trends[0][0]))))
           ),
        y=alt.Y('AutoBeta:Q', title='log(AutoBeta)',scale=alt.Scale(domain=[-6, 6]), axis=alt.Axis(labelColor='skyblue',titleColor='skyblue')),
        tooltip=['# of tasks', 'AutoBeta']
    )
 
    # Combine the charts with independent scales for y-axes
    chart = alt.layer(acc_chart, loss_chart).resolve_scale(y='independent').properties(
        width=500,
        height=400,
        title=title
    )

    return chart