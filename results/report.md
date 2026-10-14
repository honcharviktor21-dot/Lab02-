# Comprehensive Laboratory Report: Neural Network Implementation from Scratch
**Author:** Viktor Honchar  
**Student ID:** 72014  
**Institution:** Vistula University | Data Science Program

## 1. Architectural Overview and Initialization
This laboratory exercise implements a standard Multilayer Perceptron (MLP) explicitly designed for classifying the 64-dimensional arrays of the `load_digits` dataset into 10 distinct digit classes. The architecture utilizes two hidden layers, configured with a dynamic width parameter ($W$), utilizing Rectified Linear Unit (ReLU) activations, and concluding with a Softmax output layer.

The weight matrices are initialized using a normal distribution of $\mathcal{N}(0, 0.01)$, while bias vectors are initialized to zero. This ensures the initial outputs are roughly uniform, preventing vanishing gradients at the very first forward pass.

**Matrix Dimensionality for $W=64$:**
| Layer / Parameter | Shape Specification | Activation Function |
| :--- | :--- | :--- |
| **Input ($X$)** | `(Batch_Size, 64)` | None |
| **Hidden 1 ($W_1, b_1$)** | `(64, W)`, `(W,)` | ReLU: $f(x) = \max(0, x)$ |
| **Hidden 2 ($W_2, b_2$)** | `(W, W)`, `(W,)` | ReLU: $f(x) = \max(0, x)$ |
| **Output ($W_3, b_3$)** | `(W, 10)`, `(10,)` | Softmax |

## 2. Forward Propagation and Theoretical Loss Validation
The forward propagation phase systematically applies dot products and non-linearities. The final Softmax layer converts the raw logits into a normalized probability distribution. 

To validate the structural integrity of the untrained network, the initial Cross-Entropy loss is calculated via the formula:
$$L = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

*   **Observed Initial Loss:** $2.3026$
*   **Theoretical Expectation:** For 10 classes, an untrained network should output approximately equal probabilities for each class ($P = 0.1$). The expected negative log-likelihood is $-\ln(0.1) \approx 2.30258$. 
*   **Conclusion:** The exact match confirms the forward pass, softmax normalization, and loss function are correctly scaling inputs without mathematical overflow.

## 3. Backpropagation and Gradient Verification
Backpropagation was implemented utilizing the chain rule to calculate the partial derivatives of the loss with respect to each weight and bias. 

Before proceeding to the training loop, analytical gradients were subjected to numerical gradient checking. By perturbing a random subset of 10 weights by $\epsilon = 1e-5$, the numerical slope was compared against the derived analytical slope. All evaluated weights yielded a relative error strictly bounded below $1e-6$, proving the backpropagation calculus is mathematically sound.

## 4. Hyperparameter Evaluation and Network Convergence
The network was trained over 50 epochs utilizing Stochastic Gradient Descent (SGD) with a mini-batch size of 32. Two structural widths ($W \in \{16, 64\}$) and three learning rates ($\alpha \in \{0.001, 0.01, 0.1\}$) were tested.

### Empirical Training Results
| Hidden Width ($W$) | Learning Rate ($\alpha$) | Final Test Accuracy | Convergence Behavior |
| :---: | :---: | :---: | :--- |
| 16 | 0.001 | **10.19%** | Severe Underfitting (Stagnant)[cite: 3] |
| 16 | 0.01 | **10.19%** | Severe Underfitting (Stagnant)[cite: 3] |
| 16 | 0.1 | **51.48%** | Partial Convergence; capacity too low. |
| 64 | 0.001 | **10.19%** | Severe Underfitting (Stagnant)[cite: 3] |
| 64 | 0.01 | **10.19%** | Severe Underfitting (Stagnant)[cite: 3] |
| **64** | **0.1** | **86.85%** | **Optimal Convergence**; highest capacity. |

### Analysis of Training Dynamics
1.  **The "10.19%" Threshold:** A consistent accuracy of 10.19% indicates that the network is essentially guessing blindly. Since there are 10 digit classes, random chance yields a $\sim10\%$ success rate. As visualized in the loss curves[cite: 3], learning rates of 0.001 and 0.01 lack the magnitude required to shift the weights away from their random initializations within just 50 epochs.
2.  **Impact of Network Width:** At the functional learning rate of $\alpha = 0.1$, the wider network ($W=64$) vastly outperforms the narrow network ($W=16$). The $W=64$ architecture possesses significantly more learnable parameters, allowing it to capture the complex spatial pixel relationships of the digit matrices, climbing from a 51.48% accuracy up to 86.85%.

## 5. Computational Complexity and Execution Timing
To understand the hardware performance characteristics, both forward and backward propagation passes were timed in isolation over a median of 100 benchmark iterations.

### Microsecond Benchmark Metrics
| Operation | Median Execution Time | Relative Cost |
| :--- | :--- | :--- |
| **Forward Pass** | $0.0403 \text{ ms}$ | $1.0\times$ (Baseline)[cite: 4] |
| **Backward Pass** | $0.0597 \text{ ms}$ | $1.48\times$ (Slower)[cite: 4] |

### Matrix Operation Analysis
The backward pass exhibits a ~48% computational penalty compared to the forward pass[cite: 4]. During forward propagation, the algorithm only executes standard matrix multiplications: $Z^{[l]} = A^{[l-1]} \cdot W^{[l]}$.

However, backpropagation requires calculating two distinct sets of gradients per layer:
1.  **Gradients of the weights:** $dW^{[l]} = (A^{[l-1]})^T \cdot dZ^{[l]}$
2.  **Gradients of the upstream activations:** $dA^{[l-1]} = dZ^{[l]} \cdot (W^{[l]})^T$

The necessity to constantly compute matrix transposes (e.g., $(A)^T$ and $(W)^T$) and process double the amount of dot products per layer inherently makes the backward pass the dominant factor in the algorithm's time complexity[cite: 4].
