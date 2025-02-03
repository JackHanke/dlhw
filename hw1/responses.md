# Homework 1 Deep Learning
Dan, Nicole, & Jack

## Question 1

Call the data in the table $x_1$ through $x_5$.

### Perceptron (zero-one) loss:
a. $x_1$ and $x_2$ have a perceptron loss of $1$, as the prediction is the different sign as the ground truth.

b. $x_3, x_4, x_5$ all have a perceptron loss of $0$, as the prediction is the same sign as the ground truth.

c. We certainly can choose weights that yield a lower total perceptron loss. We would make our decision boundary look like the equation $x=3$.

d. No, because the loss function's gradient is $0$ everywhere except the decision boundary, so training a multilayer perceptron with gradient descent would result in no learning.

### Squared error loss:
a. $x_5$ has the highest loss value, as it has the greatest distance from its prediction to the ground truth.

b. Similarly $x_3$ has the lowest loss value, as it has the smallest distance from its prediction to the ground truth. 

c. Maybe, as the scale of $g$ affects the loss, but the decision boundary only plots the sign of $g$. Moving the decision boundary closer to $x_5$ might yield a better total loss by noticeably decreasing the large $x_5$ loss at the expense of slightly increasing the other losses. 

d. No, as this loss function for a classification task penalizes properly classified data points that just so happen to be far away from the decision boundary. For example, $x_5$ is classified correctly, and yet contributes to over 70% of the total loss. This is because the squared function is symmetric, and so significant deviations in either direction are penalized. This would lead to either bad performance or slow learning in a perceptron.

### Binary cross-entropy loss:
a. $x_2$ has the highest loss, as it is most misclassified, as it is large and positive while the ground truth is negative.

b. $x_5$ has the lowest loss, as it is very strongly positive when the ground truth is positive.

c. Yes, a similar boundary to the one described for the perceptron loss would almost certainly result in a smaller BCE.

d. Yes, BCE would be a good choice for training an MLP for classification, because it heavily penalizes misclassification while not penalizing proper classification as much.

### Hinge loss:
a. $x_2$ has the highest loss, as it is most misclassified, as it is large and positive while the ground truth is negative.

b. $x_3, x_5$ are both classified correctly and greater in absolute value than $1$, and so receive a loss of $0$.

c. Yes, a similar boundary to the one described for the perceptron loss would almost certainly result in a smaller BCE.

d. Yes, Hinge would be a good choice for training an MLP for classification, because it heavily penalizes misclassification while not penalizing proper "confident" classification at all.

## Question 2



## Question 3
### a. 

The *run_one_epoch* function is responsible for training and evaluation in a single epoch. It starts with whether the model is in training or evaluation mode. If training, it clears the gradients and then processes the input data, producing an output tensor of shape *(N, 2)*, where *N* represents the batch size, and each row contains two logits corresponding to the two possible classes in the classification task. 

To compute loss, the function applies *torch.nn.CrossEntropyLoss*, which uses softmax internally to normalize the logits into probability values before calculating the log loss. The function then determines accuracy by selecting the class with the highest logit using *torch.argmax(output, dim = 1)* and comparing it with the true class labels. Since the target labels are provided in binary format with values of either 1 or -1, the function standardizes them to 0 and 1 using *torch.where(y > 0, 1, 0)*. Lastly, the accuracy is computed by the proportion of correctly classified samples.

If training, the function performs backpropagation by calling *loss.backward()*, allowing gradients to propagate through the network. The optimizer then updates the parameters using the gradients. The reason the model outputs a tensor of shape *(N, 2)* instead of a single value per sample is that *CrossEntropyLoss* expects logits for all possible classes to correctly calculate the gradients and to optimize learning. This allows for better numerical stability and more effective (and efficient) learning, particularly in multi-class classification.

### b. 

The *run_experiment* function oversees the entire training and evaluation process by calling *pretrain_and_train*, which runs the model on an easier dataset before fine-tuning with the main dataset. The *plot_results* function generates six panels to visualize the different aspects of model training. 

The first panel shows the *training loss over epochs*, which tracks how well the model minimizes the objective function on the training set. A slowly decreasing curve means effective learning, while large fluctuations or staying steady might indicate an improper learning rate. 

The second panel shows the *validation loss over epochs*, showing insights into how well the model generalizes to new (unseen) data. If validation loss starts increasing while training loss continues decreasing, it indicates overfitting.

The third panel shows the *training accuracy over time*, which shows how well the model classifies training data as learning imrpoves. A rapid increase in training accuracy tells us that the model is quickly learning patterns in the dataset. 

The fourth panel shows *validation accuracy*, which we use to assess generalization. A large gap between training and validation accuracy shows overfitting, whereas a gradual increase in both shows stable learning.

The fifth panel shows the *decision boundary* learned by the model. This plot helps in understanding how well the classifier separates different classes in the feature space. A well-defined boundary with minimal misclassified points suggests effective learning.

The final panel shows *test set classification results*, showing the final performance of the trained model on unseen data. This panel evaluates the generalization ability of the model and determines whether it is good enough for deployment.

## Question 4


