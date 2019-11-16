import pandas as pd

# Load Data

table_1 = pd.read_csv('./user_apps_statistics.csv')
table_2 = pd.read_csv('./user_purchase_events.csv')
table_3 = pd.read_csv('./user_table.csv')

# process
table_1['ShopAppRatio'] = table_1['n_shoppingApps']/table_1['nTotal_Apps']

# combine
test = table_1[['ShopAppRatio','user_id']]\
    .merge(table_3[['gender','source_id','country_id','bin_age','user_id']], on='user_id', how='outer')

train = test.merge(table_2[['amount_spend','user_id']].replace("rookie",1).replace("casual",3).replace("player",5)
                   .replace("whale",10), on='user_id', how='outer')
train['amount_spend'].fillna(0,inplace=True)

# output
filename_1 = 'train.csv'
filename_2 = 'test.csv'

train.to_csv(filename_1,index=True)
test.to_csv(filename_2,index=True)


