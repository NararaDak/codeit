import numpy as np
import matplotlib.pyplot as plt

test = np.array([
    [0,0,1,0,0],
    [0,1,1,0,0],
    [1,0,1,1,0],
]).astype("uint8")
#plt.figure(figsize=(3,3))
# plt.imshow(test)#,cmap='gray',vmin=0,vmax=0)
# plt.show()

def visualize_tensor(tensor, title="Tensor Visualization"):
    """
    Visualizes a 3D tensor as a series of 2D slices.

    Parameters:
    tensor (np.ndarray): A 3D numpy array to visualize.
    title (str): Title for the visualization window.
    """
    if tensor.ndim != 3:
        raise ValueError("Input tensor must be a 3D numpy array.")

    depth = tensor.shape[0]
    fig, axes = plt.subplots(1, depth, figsize=(depth * 3, 3))
    fig.suptitle(title)

    for i in range(depth):
        axes[i].imshow(tensor[i], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Slice {i}')
        axes[i].axis('off')

    plt.show()

