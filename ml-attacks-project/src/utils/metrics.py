def accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return correct / total

def loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()  # Mean Squared Error as an example