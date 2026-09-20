# ML_practice_interviews

Colab notebooks to practice basic PyTorch interview questions InshaAllah

## Notebook Questions

### 1. Basic_Linear_Regression
Your task is to implement a Linear Regression model using PyTorch. The model should predict a continuous target variable based on a given set of input features.

#### Requirements

* **Model Definition:** 
  Implement a class `LinearRegressionModel` with a single linear layer mapping input features to the target variable.
* **Forward Method:** 
  Implement the `forward` method to compute predictions given input data.

---

### 2. Custom_Dataset_and_DataLoader
You are tasked with creating a custom Dataset and Dataloader in PyTorch to load data from a given `data.csv` file. The loaded data will be used to run a pre-implemented linear regression model.

#### Requirements

* **Dataset Class (`CustomDataset`):**
  * Reads data from a provided `data.csv` file.
  * Stores the features ($X$) and target values ($Y$) separately.
  * Implements PyTorch's `__len__` and `__getitem__` methods for indexing.
* **Dataloader:**
  * Use PyTorch's `DataLoader` to create an iterable for batch loading the dataset.
  * Support user-defined batch sizes and shuffling of the data.

---

### 3. Custom_Activation
You are tasked with implementing a custom activation function in PyTorch that computes the operation: $\tanh(x) + x$.

Once implemented, this custom activation function will be used in a simple linear regression model.

#### Requirements

* **Custom Activation Function:**
  * Implement a class inheriting from `torch.nn.Module`.
  * Define the `forward` method to compute the activation function ($\tanh(x) + x$).
* **Integration with Linear Regression:**
  * Use the custom activation function in a simple linear regression model.
  * The model should include a single linear layer with the custom activation function applied to its output.
* **Constraints:**
  * The custom activation function should not have any learnable parameters.
  * Ensure compatibility with PyTorch tensors for forward pass computations.

---

### 4. HuberLoss
You are tasked with implementing the Huber Loss as a custom loss function in PyTorch. The Huber loss is a robust loss function used in regression tasks, less sensitive to outliers than Mean Squared Error (MSE). It transitions between L2 loss (squared error) and L1 loss (absolute error) based on a threshold parameter.

#### Requirements

* **Custom Loss Function:**
  * Implement a class `HuberLoss` inheriting from `torch.nn.Module`.
  * Define the `forward` method to compute the Huber loss as per the formula.
* **Usage in a Regression Model:**
  * Integrate the custom loss function into a regression training pipeline.
  * Use it to compute and optimize the loss during model training.
* **Constraints:**
  * The implementation must handle both scalar and batch inputs for true values ($y$) and predicted values ($\hat{y}$).
 
### 5. Custom DNN ( Deep Neural Network)
You are tasked with constructing a Deep Neural Network (DNN) model to solve a classification/regression task using PyTorch. The objective is to predict target values from synthetic data exhibiting a non-linear relationship.

#### Requirements

* **Implement the DNNModel class that satisfies the following criteria:**
  * An input layer connected to a hidden layer.
  * A ReLU activation function for non-linearity.
  * An output layer with a single unit for regression.
  * Experiment with different numbers of layers and hidden units to optimize performance.
  * Ensure the final layer has a single output unit

### 6. TensorBoard 
You are tasked with using TensorBoard to monitor the training progress of a linear regression model in PyTorch. TensorBoard provides a visual interface to track metrics such as loss during training, making it easier to analyze and debug your model.

### Requirements

* **TensorBoard Integration**:

  * Set up TensorBoard using torch.utils.tensorboard.SummaryWriter.
  * Log the training loss after each epoch.

* **Visualization**:
  
  * Start TensorBoard using the command:
  * tensorboard --logdir=runs
  * Visualize the loss curve during training.
  * Constraints
  
  * Ensure that TensorBoard logs are saved in a directory named runs.
  * The solution must handle multiple epochs and log the loss consistently.
  * Save the model and load the model from the saved path
  
  
