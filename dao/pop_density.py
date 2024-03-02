import pymysql
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st







def get_data_from_database():
    connection = pymysql.connect(host='localhost', port=3306, user='root', passwd='',
                                 db='pop_density', charset='utf8', autocommit=True)

    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM whole_data"
            cursor.execute(sql)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=['year', 'city', 'pop', 'school', 'academy', 'transport', 'APT',
                                               'receipts', 'company', 'hospital'])
    finally:
        connection.close()

    return df

def train_model(df):
    COL_NUM = ['pop']
    COL_Y = ['school', 'academy', 'transport', 'APT', 'receipts', 'company', 'hospital']

    proj = df.drop(['year', 'city'], axis=1)

    X_tr, _, y_tr, _ = train_test_split(proj[COL_NUM], proj[COL_Y], test_size=0.3)

    scaler = StandardScaler()
    scaler.fit(X_tr[COL_NUM])
    X_tr[COL_NUM] = scaler.transform(X_tr[COL_NUM])

    new_rf_regressor = RandomForestRegressor(random_state=42, criterion='squared_error')
    params = {
    'n_estimators': st.randint(100,500),
    'max_depth': st.randint(10,50)
     }
    # GridSearchCV 객체 생성
    grid_search = RandomizedSearchCV(new_rf_regressor, params, n_jobs = -1)


    # 그리드 탐색 수행
    multioutput_model = MultiOutputRegressor(grid_search)
    multioutput_model.fit(X_tr, y_tr)
    return multioutput_model

if __name__ == '__main__':
    df = get_data_from_database()
    model = train_model(df)
    joblib.dump(model, 'pop.pkl')