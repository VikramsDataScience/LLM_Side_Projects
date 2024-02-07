# Shopping Cart Recommender Model (POC Complete)
This project is a recommender model based on ecommerce orders data.

## On Neural Collaborative Filtering
Whilst more traditional Matrix Factorisation techniques are adept at establishing linear relationships between variables by decomposing a user-item interaction matrix into the product of two lower-dimensional matrices, representing latent factors for users and items:<br>

$$R \approx P.Q^T$$

Where:
- $P$ is a matrix of user latent factors, where each row $p_u$ corresponds to the latent factors for user $u$.
- $Q$ is a matrix of item latent factors, where each row $q_i$
  corresponds to the latent factors for item $i$
- $Q^T$ is the transpose of matrix $Q$

Multi Layer Perceptrons are adept at establishing non-linear relationships (by way of the summed dot product between an input and weight + bias):<br> 

$$y=f(\sum_{i=1}^n(x_i.w_i)+b)$$<br>

Where:
- $y$ is the output of the perceptron
- $f$ is the activation function
- $x_i$ are the input values
- $w_i$ are the weights associated with the inputs
- $b$ is the bias term

The Neural Collaborative Filtering (NCF) technique is a hybrid between the aforementioned methods. In this respect, NCF can establish both linear and non-linear dynamical relationships between variables:

$$f(u, i) = \sigma(\text{MLP}(\text{Concatenate}(u, i)))$$

Where:
- $f(u, i)$: This is the output of the network, which is a function of the user embedding and the item embedding.
- $\sigma()$: This is the sigmoid activation function, which is a non-linear function that squashes the input values to the range [0, 1].
- $\text{MLP}()$: This is the multi-layer perceptron function, which is a type of neural network with multiple hidden layers.
- $\text{Concatenate}()$: This is the concatenation function, which combines the user embedding and the item embedding into a single vector.

## Module Usage
Please run the modules in the following order. There aren't any args that need to be parsed through the command line, so running each module can be run using `python <module_name>`. To change any args, please use the `config.yml` file to change args as required:
1. `Preprocessing_EDA`: Running this module will generate a `ydata profiling` report that can be opened in a browser as an HTML file. Furthermore, this module will generate a sparse coordinate matrix for the requisite data points - 'order_id', 'reordered' (the flag that is used as a substitute for ratings in a traditional Recommender), and 'product_id'.
2. `NCF_Model_Train`: Running this module will train an NCF model and save the trained model to a storage location (the model should take ~3 hours to train on a decent GPU). Regarding training time, after this model was trained, I did discover a Learning Rate Finder (https://pytorch-lightning.readthedocs.io/en/1.4.9/advanced/lr_finder.html) that is said to be able to calculate the most optimal learning rates during backpropogation. I haven't implemented or tested this, but instead I had defined the LRs by manualing defining the 'step_size' and 'gamma' values (i.e. step_size x gamma = epochs).<br>
&nbsp;&nbsp;The boilerplate code that defines the NCF model architecture, as described in the above equations, has been written as a class with two methods in the `NCF_Architecture_config.py` file in PyTorch. This file gets loaded into the module upon running it.
3. `NCF_Model_Evaluate`: This module will calculate `MAE`, `RMSE`, and `Average loss`.