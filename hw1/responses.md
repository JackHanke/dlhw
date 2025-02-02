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



## Question 4


