import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


# Define the function for splitting data into train, dev, and test sets
def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), shuffle=False)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=(dev_size / (test_size + dev_size)),
                                                    shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


# Define the function for predicting and evaluating the model
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    print(f"Classification report for classifier {model}:\n{metrics.classification_report(y_test, predicted)}\n")

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print("Classification report rebuilt from confusion matrix:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")


# Load the digits dataset
digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))

# Split data into train, dev, and test sets
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data, digits.target, test_size=0.5, dev_size=0.2)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict and evaluate the classifier on the dev set
predict_and_eval(clf, X_dev, y_dev)

# Visualize the dev set predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_dev, clf.predict(X_dev)):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

plt.show()
