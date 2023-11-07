from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from sklearn import datasets, svm, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from skimage.transform import resize

app = Flask(__name)

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

@app.route('/api/upload_images', methods=['POST'])
def upload_images():
    uploaded_files = request.files.getlist('image')
    if len(uploaded_files) != 2:
        return jsonify({"error": "Please provide exactly two images."})

    digit_predictions = []
    for file in uploaded_files:
        try:
            image = Image.open(file)
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((8, 8))  # Resize to 8x8
            data = np.array(image).flatten()
            digit_predictions.append(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    X = np.array(digit_predictions)
    
    # Split the data for model evaluation
    test_size = 0.2
    dev_size = 0.1
    train_size = 1.0 - test_size - dev_size
    X_train, X_dev, X_test, y_train, y_dev, y_test = Split_Train_Dev_Test(digits.data, digits.target, test_size, dev_size)

    # SVM Model
    svm_model = svm.SVC()
    svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.001, 0.01, 0.1]}
    svm_best_hparams, svm_best_model, svm_best_accuracy = tune_hparams(svm_model, svm_param_grid, X_train, y_train, X_dev, y_dev)

    # Decision Tree Model
    tree_model = tree.DecisionTreeClassifier()
    tree_param_grid = {'max_depth': [5, 10, 15]}
    tree_best_hparams, tree_best_model, tree_best_accuracy = tune_hparams(tree_model, tree_param_grid, X_train, y_train, X_dev, y_dev)

    svm_prediction = svm_best_model.predict(X)
    tree_prediction = tree_best_model.predict(X)

    return jsonify({"svm_prediction": int(svm_prediction[0]), "tree_prediction": int(tree_prediction[0])})
# the application 
@app.route('/api/compare_images', methods=['POST'])
def compare_images():
    data = request.get_json()
    if 'svm_prediction' in data and 'tree_prediction' in data:
        svm_prediction = data['svm_prediction']
        tree_prediction = data['tree_prediction']
        result = svm_prediction == tree_prediction
        return jsonify({"result": result})
    else:
        return jsonify({"error": "Please provide both SVM and Decision Tree predictions."})

if __name__ == '__main__':
    app.run(debug=True)
