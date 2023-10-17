import random
from math import exp


# "AND" Dataset
train_data = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 1),
]

# "OR" Dataset
# train_data = [
#     (0, 0, 0),
#     (1, 0, 1),
#     (0, 1, 1),
#     (1, 1, 1),
# ]

train_count = len(train_data)


# Define the sigmoid function for activation.
def sigmoid(x):
    return 1 / (1 + exp(-x))


# Define the cost function to evaluate the model's performance.
def cost(w1, w2, b):
    result = 0
    for i in range(train_count):
        x1 = train_data[i][0]
        x2 = train_data[i][1]
        y = sigmoid(x1 * w1 + x2 * w2 + b)
        d = y - train_data[i][2]
        result += d ** 2

    result /= train_count

    return result


# Define the main training loop.
def main():
    random.seed(0)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    b = random.uniform(0, 1)

    eps = 1e-3
    rate = 1e-3

    for epoch in range(1000000):
        c = cost(w1, w2, b)

        # Calculate the partial derivatives of the cost function with respect to weights and bias using finite differences.
        dw1 = (cost(w1 + eps, w2, b) - c) / eps
        dw2 = (cost(w1, w2 + eps, b) - c) / eps
        db = (cost(w1, w2, b + eps) - c) / eps

        # Update weights and bias using gradient descent.
        w1 -= rate * dw1
        w2 -= rate * dw2
        b -= rate * db

        # if epoch % 1000 == 0:
        #     print(w1, w2, b, cost(w1, w2, b))

    # Print the trained model's predictions for the training data.
    for i in range(train_count):
        print(train_data[i][0], train_data[i][1], sigmoid(train_data[i][0] * w1 + train_data[i][1] * w2 + b))


# Call the main function to start training and testing the model.
main()
