# Bioinformática y Estadística II - IV Clasificación Automática 
## Mushroom Classification
### Varela Vega Alfredo 

import pandas 

# Reading the csv file 
df_mushroom = pandas.read_csv("data/mushrooms.csv")
print(df_mushroom)



# Transforming letter labels to numeric labels for applying the models

from sklearn import preprocessing 

# Shortcut for calling the encoder 
le = preprocessing.LabelEncoder()

def transform_columns(category):
    '''
    desc: transform non numerical labels to numerical labels
    '''
    return le.fit_transform(category) 




for col in df_mushroom.columns:
    df_mushroom[col]= transform_columns(df_mushroom[col])
    
print(df_mushroom)

# Spliting dataset into random train and test subsets

from sklearn.model_selection import train_test_split 

# Reading al categories but class from 1 to end
X= df_mushroom.drop(["class"],axis=1)
# Reading classes (edible vs poisonous)
y= df["class"] 

# Spliting dataset in training (70%) and evaluation (30%) using 0 as a seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Classifing by k-nearest neighbor 

from sklearn.neighbors import KNeighborsClassifier

# Setting the number of nearest neighbors 
k=1

# Classifier's definition
classifier = KNeighborsClassifier(n_neighbors=k)
# Training the classifier with training dataset and class values 
classifier.fit(X_train, y_train)
# Prediction of the evaluation dataset using trained classifier 
y_predict = classifier.predict(X_test) 
print(y_predict)



## Classifier evaluation 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Accuracy: {}".format(accuracy_score(y_test,y_predict)))
print("Precision:{}".format(precision_score(y_test, y_predict, average="macro" )))
print("recall:{}".format(recall_score(y_test, y_predict, average="macro")))
print("F-score:{}".format(f1_score(y_test, y_predict, average="macro")))

from sklearn.metrics import classification_report

target_names= ["edible","poisonous"]
print(classification_report(y_test, y_predict,target_names=target_names))

from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix 

print(confusion_matrix(y_test, y_predict))
plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues, display_labels=["edible", "poisonous"])

# Classifing by Support Vector Machine

from sklearn.svm import SVC
# Classifier's Definiton 
svm_classifier= SVC(kernel="linear")
# Training classifier with training dataset and training values 
svm_classifier.fit(X_train, y_train)
# Classifier's prediction with evaluation dataset
y_predict= svm_classifier.predict(X_test)
print(y_predict)

## Classifier evaluation 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Accuracy: {}".format(accuracy_score(y_test, y_predict)))
print("Preceision: {}".format(precision_score(y_test, y_predict, average="macro")))
print("Recall {}".format(recall_score(y_test, y_predict, average="macro")))
print("F-score {}".format(f1_score(y_test, y_predict, average="macro")))


from sklearn.metrics import classification_report 

# Setting class labels
target_names= ["edible","poisonous"]
print(classification_report(y_test, y_predict, target_names= target_names))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix

print(confusion_matrix(y_test, y_predict))
plot_confusion_matrix(svm_classifier, X_test, y_test, cmap=plt.cm.Blues, display_labels=["edible", "posionous"])

