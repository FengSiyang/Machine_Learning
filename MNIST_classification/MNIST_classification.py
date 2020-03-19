"""
The problem of MNIST number classification
"""

from Read_MNIST import read_mnist, plot_number, plot_numbers

train_x, train_y, test_x, test_y = read_mnist()

# change pixel represent from [0-255] to [0, 1]:
train_x = train_x.reshape(-1, 28 * 28).astype(float) / 255  # change [28, 28] to [784], each sample for training
test_x = test_x.reshape(-1, 28 * 28).astype(float) / 255



'''
================== Classification:
Aim to make test_y and real_y more closer.
Loss Function: testing the level of closest of real_y and test_y.
'''

#--------------- linear classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def logist_regression(train_x, train_y, test_x):
    # multi classification in logistic regression -> softmax regression
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    ### Use 'pickle', 'pickle.dump' to save the trained model for the further usage  
    lr_y = lr.fit(train_x, train_y).predict(test_x)
    return lr_y

#lr_y = logist_regression(train_x, train_y, test_x)
#print(accuracy_score(test_y, lr_y))



#------------------ KNN
from sklearn.neighbors import KNeighborsClassifier
def knn_classify(train_x, train_y, test_x):
    knn = KNeighborsClassifier()
    knn_y = knn.fit(train_x, train_y).predict(test_x)
    return knn_y

#knn_y = knn_classify(train_x, train_y, test_x)
#print(accuracy_score(test_y, knn_y))




#------------------ Decision Tree
from sklearn.tree import DecisionTreeClassifier
def dt_classify(train_x, train_y, test_x):
    dt = DecisionTreeClassifier()
    dt_y = dt.fit(train_x, train_y).predict(test_x)
    return dt_y

#dt_y = dt_classify(train_x, train_y, test_x)
#print(accuracy_score(test_y, dt_y))



#---------------- SVM: (very slow in multi classification problem)
from sklearn.svm import SVC
def svm_classify(train_x, train_y, test_x):
    svm = SVC()
    svm_y = svm.fit(train_x, train_y).predict(test_x)
    return svm_y




"""
Random Forest and AdaBoost:
merge some weak models into one stronger model
"""


#--------------- Neural network
from sklearn.neural_network import MLPClassifier  # Multilayer Perceptron
def nn_classify(train_x, train_y, test_x):
    mlp = MLPClassifier()
    mlp_y = mlp.fit(train_x, train_y).predict(test_x)
    return mlp_y



#---------------- PCA & Pipeline:
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def pca_pipeline(train_x, train_y, test_x):
    pca = PCA(n_components=50)
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    pca_lr = Pipeline([('pca1', pca), ('lr2', lr)])
    pca_lr_y = pca_lr.fit(train_x, train_y).predict(test_x)
    return pca_lr_y

pca_lr_y = pca_pipeline(train_x, train_y, test_x)
print(accuracy_score(test_y, pca_lr_y))





