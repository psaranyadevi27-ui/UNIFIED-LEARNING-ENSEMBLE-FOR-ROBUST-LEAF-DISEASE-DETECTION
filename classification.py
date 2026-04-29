from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from Confusion_matrix import multi_confu_matrix


def UnifiedLearningEnsemble(x_train, y_train, x_test, y_test):

    # Define individual classifiers
    decision_tree_clf = DecisionTreeClassifier(random_state=42)
    svm_clf = SVC(probability=True, random_state=42)
    nn_clf = MLPClassifier(random_state=42)

    # Define the ensemble classifier using weighted voting
    ensemble_clf = VotingClassifier(estimators=[
        ('decision_tree', decision_tree_clf),
        ('svm', svm_clf),
        ('neural_network', nn_clf)
    ], voting='soft', weights=[2, 1, 2])  # Adjust weights based on classifier performance

    # Train the ensemble classifier
    ensemble_clf.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = ensemble_clf.predict(x_test)

    # Calculate accuracy
    met = multi_confu_matrix(y_test, y_pred)
    return y_pred, met


def SVM(x_train, y_train, x_test, y_test):
    # Create an SVM classifier
    svm_clf = SVC(probability=True, random_state=42)

    # Train the SVM classifier
    svm_clf.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = svm_clf.predict(x_test)
    met = multi_confu_matrix(y_test, y_pred)
    return y_pred, met


def KNN(x_train, y_train, x_test, y_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Train the KNN classifier
    knn_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = knn_classifier.predict(x_test)
    met = multi_confu_matrix(y_test, y_pred)
    return y_pred, met


from sklearn.ensemble import RandomForestClassifier

def RF(x_train, y_train, x_test, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    met = multi_confu_matrix(y_test, y_pred)
    return y_pred, met

def ANN(x_train, y_train, x_test, y_test):
    ann_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    # Train the classifier on the training data
    ann_classifier.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = ann_classifier.predict(x_test)
    met = multi_confu_matrix(y_test, y_pred)
    return y_pred, met


def DTree(x_train, y_train, x_test, y_test):
    # Create a Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    # Train the Decision Tree model
    dt_model.fit(x_train, y_train)
    y_pred_dt = dt_model.predict(x_test)
    return y_pred_dt, multi_confu_matrix(y_test, y_pred_dt)

