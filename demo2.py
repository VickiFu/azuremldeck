%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import model algorithm and data
from sklearn import svm, datasets

# Import splitter
from sklearn.cross_validation import train_test_split

# Import metrics method
from sklearn.metrics import confusion_matrix

# Feature data (X) and labels (y)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size = 0.90, random_state = 42)
#1

# Perform the classification step and run a prediction on test set from above
clf = svm.SVC(kernel = 'rbf', C = 0.5, gamma = 10)
y_pred = clf.fit(X_train, y_train).predict(X_test)

pd.DataFrame({'Prediction': iris.target_names[y_pred],
    'Actual': iris.target_names[y_test]})
#2

# Accuracy score
clf.score(X_test, y_test)
#3

# Define a plotting function confusion matrices 
#  (from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
def plot_confusion_matrix(cm, target_names, title = 'The Confusion Matrix', cmap = plt.cm.YlOrRd):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.tight_layout()
    
    # Add feature labels to x and y axes
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.colorbar()
#4


#Numbers in confusion matrix:
#•on-diagonal - counts of points for which the predicted label is equal to the true label
#•off-diagonal - counts of mislabeled points
cm = confusion_matrix(y_test, y_pred)

# Actual counts
print(cm)

# Visually inpsect how the classifier did of matching predictions to true labels
plot_confusion_matrix(cm, iris.target_names)






