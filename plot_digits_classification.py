import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm, tree
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

# Define the function for splitting data into train, dev, and test sets
def Split_Train_Dev_Test(X, y, test_size, dev_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), shuffle=False)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=(dev_size / (test_size + dev_size)), shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Define the function for resizing images
def resize_images(data, new_shape):
    resized_data = []
    for image in data:
        resized_image = resize(image, new_shape, anti_aliasing=True)
        resized_data.append(resized_image)
    return resized_data

# Function for hyperparameter tuning
def tune_hparams(model, param_grid, X_train, y_train, X_dev, y_dev):
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_accuracy = best_model.score(X_dev, y_dev)
    return grid_search.best_params_, best_model, best_accuracy

# Load the MNIST dataset
digits = datasets.load_digits()

# Modify this part
image_sizes = [(4, 4), (6, 6), (8, 8)]  # Define the image sizes to evaluate

for image_size in image_sizes:
    resized_data = resize_images(digits.images, image_size)
    data = np.array(resized_data).reshape((len(resized_data), -1))

    test_size = 0.2
    dev_size = 0.1
    train_size = 1.0 - test_size - dev_size
    X_train, X_dev, X_test, y_train, y_dev, y_test = Split_Train_Dev_Test(data, digits.target, test_size, dev_size)

    # SVM Model
    svm_model = svm.SVC()
    svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.001, 0.01, 0.1]}
    svm_best_hparams, svm_best_model, svm_best_accuracy = tune_hparams(svm_model, svm_param_grid, X_train, y_train, X_dev, y_dev)

    # Decision Tree Model
    tree_model = tree.DecisionTreeClassifier()
    tree_param_grid = {'max_depth': [5, 10, 15]}
    tree_best_hparams, tree_best_model, tree_best_accuracy = tune_hparams(tree_model, tree_param_grid, X_train, y_train, X_dev, y_dev)

    print(f"Image size: {image_size[0]}x{image_size[1]} Train size: {train_size} Dev size: {dev_size} Test size: {test_size}")
    print(f"SVM Model Accuracy: {svm_best_accuracy}")
    print(f"Decision Tree Model Accuracy: {tree_best_accuracy}")

    # Confusion Matrix
    svm_predictions = svm_best_model.predict(X_test)
    tree_predictions = tree_best_model.predict(X_test)
    confusion = confusion_matrix(svm_predictions, tree_predictions)

    print("Confusion Matrix:")
    print(confusion)

    # Calculate the F1 score for each class and report macro-average F1
    f1_svm = f1_score(y_test, svm_predictions, average=None)
    f1_tree = f1_score(y_test, tree_predictions, average=None)
    macro_avg_f1_svm = f1_score(y_test, svm_predictions, average='macro')
    macro_avg_f1_tree = f1_score(y_test, tree_predictions, average='macro')

    print(f"SVM Model F1 Scores: {f1_svm}")
    print(f"Decision Tree Model F1 Scores: {f1_tree}")
    print(f"Macro-Average F1 for SVM Model: {macro_avg_f1_svm}")
    print(f"Macro-Average F1 for Decision Tree Model: {macro_avg_f1_tree}")

    # Bonus: Confusion matrix for samples predicted correctly by production but not by the candidate
    correct_svm = (svm_predictions == y_test)
    correct_tree = (tree_predictions == y_test)

    correct_in_svm_not_tree = np.logical_and(correct_svm, np.logical_not(correct_tree))
    correct_in_tree_not_svm = np.logical_and(correct_tree, np.logical_not(correct_svm))

    bonus_confusion = np.array([[np.sum(correct_in_svm_not_tree), np.sum(correct_in_svm_not_tree)],
                                [np.sum(correct_in_tree_not_svm), np.sum(correct_in_tree_not_svm)]])

    print("Bonus: Confusion Matrix for Samples Predicted Correctly by Production but Not by Candidate:")
    print(bonus_confusion)
