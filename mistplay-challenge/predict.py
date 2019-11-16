import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# Load Data
table_1 = pd.read_csv('./user_apps_statistics.csv')
table_2 = pd.read_csv('./user_purchase_events.csv')
table_3 = pd.read_csv('./user_table.csv')

# process
table_1['ShopAppRatio'] = table_1['n_shoppingApps']/table_1['nTotal_Apps']

# combine
test_data = table_1[['ShopAppRatio','user_id']]\
    .merge(table_3[['gender','source_id','country_id','bin_age','user_id']], on='user_id', how='outer')

# Only in binary, needs improve
train_data = test_data.merge(table_2[['amount_spend','user_id']].replace("rookie",1).replace("casual",1).replace("player",1)
                   .replace("whale",1), on='user_id', how='outer')
train_data['amount_spend'].fillna(0,inplace=True)

# output
train_data.to_csv('train_data.csv',index=True)
test_data.to_csv('test_data.csv',index=True)

# Use mean ShopAppRatio to fill nan value
train_data['ShopAppRatio'].fillna(train_data['ShopAppRatio'].mean(), inplace=True)
test_data['ShopAppRatio'].fillna(test_data['ShopAppRatio'].mean(), inplace=True)

# Use 0 to fill nan gender
train_data['gender'].fillna(0, inplace=True)
test_data['gender'].fillna(0, inplace=True)

# Use source_5 to fill nan value
train_data['source_id'].fillna("source_5", inplace=True)
test_data['source_id'].fillna("source_5", inplace=True)

# Use country_0 to fill nan value
train_data['country_id'].fillna("country_1", inplace=True)
test_data['country_id'].fillna("country_1", inplace=True)

# Use (35.0, 40.0] to fill nan value
train_data['bin_age'].fillna("(35.0, 40.0]", inplace=True)
test_data['bin_age'].fillna("(35.0, 40.0]", inplace=True)

# Select features
features = ['ShopAppRatio', 'gender', 'source_id', 'country_id', 'bin_age']
train_features = train_data[features]
train_labels = train_data['amount_spend']
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
print('Score accuracy is %.4lf' % acc_decision_tree)

# Create a  DataFrame
submission = pd.DataFrame({'user_id':test_data['user_id'],'amount_spend':pred_labels})
submission.to_csv('Game Purchase Predictions.csv',index=True)
