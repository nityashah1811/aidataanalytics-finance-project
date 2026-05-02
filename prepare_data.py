import xgboost as xgb
import pandas as pd
import numpy as np


def prep(df):

    #print(df.describe())
    #fixed_cost = ['tuition']
    expenses = ['tuition', 'housing', 'food', 'transportation', 'books_supplies','entertainment','personal_care','technology','health_wellness','miscellaneous'] #this is per month
    df['total_monthly_spending'] = df[expenses].sum(axis=1)
    df['net_income'] = df['monthly_income']-df['total_monthly_spending']

    #will classify based on net_income
    df['initial_classification']=np.select(
        condlist=[df['net_income']<=0, df['net_income']>0],
        choicelist=["overspending", "making_money"],
        default="no_profit"
    )

    #will be clustering on the factors: housing, food, entertainment, total spending
        #and net_income ofc
    return df

def load_data():
    df = pd.read_csv("student_spending (1).csv")
    df1 = prep(df)
    return df1
