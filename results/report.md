 Viktor Honchar 72014

## 1. Network Architecture
A two-hidden-layer neural network was designed to process the 64-dimensional input of the `load_digits` dataset. The layers transition as follows: 
Input (64) → ReLU Hidden (Width W) → ReLU Hidden (Width W) → Softmax Output (10). 
Weights were initialized using a normal distribution $N(0, 0.01)$ and biases were initialized to zero.

## 2. Analytical vs. Numerical Gradients
The analytical gradients derived via the chain rule were successfully validated against numerical gradient checking. 10 random weights were nudged by $\pm1e-5$, and the relative error between the computed numerical slope and the analytical slope remained strictly below the required $1e-6$ threshold.

## 3. Training & Performance Summary
The network was trained over 50 epochs utilizing a mini-batch size of 32. 
*   Configurations pairing a hidden width of $W=64$ with a learning rate of $0.1$ converged the fastest and achieved the highest classification accuracy. 
*   A learning rate of $0.001$ proved too slow, resulting in underfitting within the 50-epoch limit. 
*   *Reference `results/training_log.txt` and `results/loss_curves.png` for explicit iteration data.*

## 4. Computational Timing Breakdown
Testing a 64-width network over a median of 100 runs revealed a clear performance cost discrepancy:
*   **Forward Pass:** Handled purely through sequential dot products and activation function applications.
*   **Backward Pass:** Takes significantly longer (visualized in `results/timing_chart.png`). This dominates the computational time because it requires transposing previous weight matrices and input data, calculating gradients for both the learnable parameters and the upstream layer activations, thereby inherently doubling the heavy matrix multiplication workload.