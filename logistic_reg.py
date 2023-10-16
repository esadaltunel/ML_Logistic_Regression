from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

labelencoder = LabelEncoder()

dataset = pd.read_csv("heart.csv")
dataset2 = dataset.copy()

numerical_columns = dataset2.describe().columns
object_columns = dataset2.drop(numerical_columns, axis = 1).columns

for i in object_columns:
    dataset2[i] = labelencoder.fit_transform(dataset2[i].values)
    
train, test = train_test_split(dataset2, test_size = 0.3)

x_train = train.drop("HeartDisease", axis = 1)
y_train = train.loc[:, "HeartDisease"]

x_test = test.drop("HeartDisease", axis = 1)
y_test = test.loc[:, "HeartDisease"]

model = LogisticRegression(solver="lbfgs", max_iter=1000)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
