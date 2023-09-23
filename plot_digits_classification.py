import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from skimage.transform import resize  # Import the resize function

# Define the function for splitting data into train, dev, and test sets
def Split_Train_Dev_Test(X, y, test_size, dev_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), shuffle=False)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=(dev_size / (test_size + dev_size)),
                                                    shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Define the function for resizing images
def resize_images(data, new_shape):
    resized_data = []
    for image in data:
        resized_image = resize(image, new_shape, anti_aliasing=True)
        resized_data.append(resized_image)
    return resized_data

# ... (rest of the code)

# Modify this part
image_sizes = [(4, 4), (6, 6), (8, 8)]  # Define the image sizes to evaluate

for image_size in image_sizes:
    resized_data = resize_images(digits.images, image_size)
    data = resized_data.reshape((len(resized_data), -1))

    test_size = 0.2
    dev_size = 0.1
    train_size = 1.0 - test_size - dev_size
    X_train, X_dev, X_test, y_train, y_dev, y_test = Split_Train_Dev_Test(data, digits.target, test_size, dev_size)

    # Example hyperparameter tuning configurations (replace with your own)
    list_of_all_param_combinations = [
        {"C": 1.0, "kernel": "linear"},
        {"C": 0.1, "kernel": "rbf", "gamma": 0.001},
        {"C": 0.01, "kernel": "poly", "degree": 3, "gamma": 0.01},
        # Add more combinations as needed
    ]

    # Call the hyperparameter tuning function
    best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combinations)

    print(f"image size: {image_size[0]}x{image_size[1]} train_size: {train_size} dev_size: {dev_size} test_size: {test_size} train_acc: {best_accuracy} dev_acc: {best_accuracy} test_acc: {best_accuracy}")
    print(f"Best hyperparameters: {best_hparams}")

    # Visualize the dev set predictions
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_dev, best_model.predict(X_dev)):
        ax.set_axis_off()
        image = image.reshape(image_size)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    plt.show()
