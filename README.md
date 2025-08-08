# Visualizing-Filters-of-CNN-using-TensorFlow-


This repository contains a Jupyter notebook that demonstrates how to visualize the filters of a Convolutional Neural Network (CNN) using a pre-trained VGG16 model. The notebook walks through the process of generating images that maximize the activation of a specific filter in a given layer.

## How it Works

The core idea is to find an input image that makes a particular filter "fire" as much as possible. This is achieved through an iterative optimization process:

1.  **Start with a random image:** A random image with a shape of `(96, 96, 3)` is initialized.
2.  **Forward Pass:** The random image is passed through the model up to the layer and filter we want to visualize.
3.  **Calculate Loss:** A loss function is defined as the mean of the output of the target filter. The goal is to maximize this value.
4.  **Backward Pass (Gradient Ascent):** Instead of standard gradient descent (which minimizes loss), we use gradient ascent to maximize the loss. The gradients of the loss with respect to the input image are calculated.
5.  **Update the Image:** The image is updated by adding the gradients multiplied by a learning rate. This pushes the image pixels in a direction that increases the filter's activation.
6.  **Repeat:** Steps 2-5 are repeated for a number of iterations until the image converges to a pattern that highly activates the chosen filter.

## Notebook Structure

The Jupyter notebook `Visualizing Filters of a CNN - Complete.ipynb` is structured into the following tasks:

### Task 1: Setup
-   Imports necessary libraries like `tensorflow` and `matplotlib`.

### Task 2: Downloading the Model
-   Downloads the `VGG16` model pre-trained on the `imagenet` dataset, but without the top classification layers (`include_top=False`).
-   Sets the input shape to `(96, 96, 3)`.
-   Prints a summary of the model architecture.

### Task 3: Get Layer Output
-   Defines a function `get_submodel` that creates a new model whose output is the activation of a specified layer. This allows us to inspect the intermediate layers of the VGG16 model.

### Task 4: Image Visualization
-   Provides a utility function `plot_image` to display an image after normalizing its pixel values for better visualization.
-   Generates and plots a random initial image to demonstrate the function.

### Task 5: Training Loop (Visualization Logic)
-   The `visualize_filter` function implements the core logic for finding the optimal image.
-   It takes `layer_name`, `learning rate`, `iterations`, and an optional `filter_index` as input.
-   It uses `tf.GradientTape` to compute gradients and updates the image iteratively.
-   Finally, it calls `plot_image` to display the resulting visualization.

### Task 6: Final Results
-   Lists all the layer names in the VGG16 model to help in selecting a layer to visualize.
-   Calls the `visualize_filter` function for a chosen layer (e.g., `'block4_conv2'`) to display the output.

## How to Run

1.  Make sure you have a Python environment with `tensorflow` and `matplotlib` installed.
2.  Open the `Visualizing Filters of a CNN - Complete.ipynb` notebook in a Jupyter environment.
3.  Run all the cells in the notebook sequentially.

This will download the VGG16 model, and for the specified layer and filter, it will generate and display an image that maximally activates that filter.
