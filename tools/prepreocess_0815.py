import pandas as pd
from graph_model.utils import *

features = ['table_schema', 'table_name', 'start', 'end', 'dep_tab_schema', 'dep_table_name', 'dep_start',
            'dep_end']

f1 = "../dataset/20200710/cloud20210710_eng.csv"
f2 = "../dataset/20210815/new_eng.csv"


df = pd.read_csv(f1, encoding='utf8')[features]
df_new = pd.read_csv(f2, encoding='utf8')[['table_schema', 'table_name', 'dep_tab_schema', 'dep_table_name',
                                           'dep_start', 'dep_end']]
print(df_new.shape)

df[['table_schema', 'table_name', 'dep_tab_schema', 'dep_table_name']] = \
    df[['table_schema', 'table_name', 'dep_tab_schema', 'dep_table_name']].apply(
        lambda x: x.astype(str).str.lower())
df_new[['table_schema', 'table_name', 'dep_tab_schema', 'dep_table_name']] = \
    df_new[['table_schema', 'table_name', 'dep_tab_schema', 'dep_table_name']].apply(
        lambda x: x.astype(str).str.lower())

table_info = df[features[:4]].reset_index(drop=True)
dep_table_info = df[features[4:]].reset_index(drop=True)
dep_table_info.columns = features[:4]
table_info = pd.concat((table_info, dep_table_info))
print("shape after concat: ", table_info.shape)
table_info = table_info.drop_duplicates(subset=['table_schema', 'table_name'])

print(table_info.head())

df_renew = df_new.merge(table_info, how='left', on=['table_schema', 'table_name'])
print(df_renew.shape)
df_renew.to_csv("../dataset/20210815/table_preprocessed_0815.csv", index=False)