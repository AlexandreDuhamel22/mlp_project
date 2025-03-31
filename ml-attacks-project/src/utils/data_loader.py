def load_data(train_path, test_path):
    import pandas as pd

    # Load training data
    train_data = pd.read_csv(train_path)
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']

    # Load test data
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    return (X_train, y_train), (X_test, y_test)