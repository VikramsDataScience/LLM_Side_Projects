# Shopping Cart Recommender Model (WIP)
This project is a recommender model based on ecommerce orders data.

## On Neural Collaborative Filtering
Whilst more traditional Matrix Factorisation techniques are adept at establishing linear relationships between variables by decomposing a user-item interaction matrix into the product of two lower-dimensional matrices, representing latent factors for users and items:<br>

$R \approx P.Q^T$

Where:
- $P$ is a matrix of user latent factors, where each row $p_u$ corresponds to the latent factors for user $u$.
- $Q$ is a matrix of item latent factors, where each row $q_i$
  corresponds to the latent factors for item $i$
- $Q^T$ is the transpose of matrix $Q$

Multi Layer Perceptions are adept at establishing non-linear relationships (by way of the summed dot product between an input and weight + bias):<br> 

$y=f(\sum_{i=1}^n(x_i.w_i)+b)$<br>

Where:
- $y$ is the output of the perceptron
- $f$ is the activation function
- $x_i$ are the input values
- $w_i$ are the weights associated with the inputs
- $b$ is the bias term

The Neural Collaborative Filtering (NCF) technique is a hybrid between the aforementioned methods. In this respect, NCF can establish both linear and non-linear dynamical relationships between variables.

$f(u, i) = \sigma(\text{MLP}(\text{Concatenate}(u, i)))$

Where:
- $f(u, i)$: This is the output of the network, which is a function of the user embedding and the item embedding.
- $\sigma()$: This is the sigmoid activation function, which is a non-linear function that squashes the input values to the range [0, 1].
- $\text{MLP}()$: This is the multi-layer perceptron function, which is a type of neural network with multiple hidden layers.
- $\text{Concatenate}()$: This is the concatenation function, which combines the user embedding and the item embedding into a single vector.