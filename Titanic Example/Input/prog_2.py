import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# Load Data
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# # View
# print(train_data.info())
# print('-'*30)
# print(train_data.describe())
# print('-'*30)
# print(train_data.describe(include=['O']))
# print('-'*30)
# print(train_data.head())
# print('-'*30)
# print(train_data.tail())

# Clean data
# Use mean age to fill nan value
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

# Use mean ticket price to fill nan value
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
print(train_data['Embarked'].value_counts())

# Use the most used port to fill nan value
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# Create ID3 Decision tree
clf = DecisionTreeClassifier(criterion='entropy')
# Train the decision tree
clf.fit(train_features, train_labels)

test_features=dvec.transform(test_features.to_dict(orient='record'))
# Predict decision tree
pred_labels = clf.predict(test_features)

# Obtain decision tree accuracy
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score accuracy is %.4lf' % acc_decision_tree)

#Display our predictions - they are either 0 or 1 for each training instance
#depending on whether our algorithm believes the person survived or not.
pred_labels

#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':pred_labels})

#Visualize the first 5 rows
submission.head()

#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)