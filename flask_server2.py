from flask import Flask, request, render_template, jsonify, redirect, make_response
import os # path dir에 관련
import json # python에서 사용하는 패키지 (한글문제때문에 이걸 써야함)
import pymysql
import numpy as np
import joblib
import traceback
import pickle
import pandas as pd
# from dao import sungjuk

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length 
# 입력하는 데이터를 검증하는거 

import logging # 로그인 , info...
from logging.config import dictConfig # ,default_handler
# cookie 사용자 make_resonse사용

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 이건 안되고 json으로 처리해야함
app.config['SECRET_KEY']= 'abcbdbdmfn'
#파일로 저장, 레벨지정

#app에 사용자 변수를 정의할 수 있음
app.config['MYSPL_HOST'] = 'localhost'
app.config['MYSPL_USER'] = 'root'
app.config['MYSPL_PORT'] = 3306
app.config['MYSPL_PASSWORD'] = ''
app.config['MYSPL_DB'] = 'sungjuk'

def getConnection():
    return pymysql.connect(
      host=app.config['MYSPL_HOST'],
      port = app.config['MYSPL_PORT'],
      user = app.config['MYSPL_USER'],
      password = app.config['MYSPL_PASSWORD'],
      db = app.config['MYSPL_DB']
)



@app.route("/") # 장식자
def index():
    # debug(상쇄), info(일반), warning(경고), error, critical(심각) 심각까지 나오면 그것만 나오고 아래는 안나모
    app.logger.info('%s logged in successfully', '성공')
    app.logger.info(f'[{request.method}] {request.path}')
    app.logger.debug("Debug log level")
    app.logger.info("Program running correctly")
    app.logger.warning("warning; low disk space!")
    app.logger.error("Error!")
    app.logger.critical("Program halt!")
    return render_template("index.html")



# 모델 불러오기
model = joblib.load("my_model.pkl")

# 입력폼으로 연결
@app.route("/insert_features")
def insert_features():
    return render_template('insert_features.html')


# 입력폼에서 입력한 변수들 예측모델 사용하여 처리 후 결과 페이지로 넘기기


@app.route('/print_predict', methods=['POST'])
def print_predict():
    if request.method == 'POST':
        #
        feature1 = request.form.get('feature1',type=float)
        feature2 = request.form.get('feature2',type=float)
        feature3 = request.form.get('feature3',type=float)
        feature4 = request.form.get('feature4',type=float)
        feature5 = request.form.get('feature5',type=float)
        feature6 = request.form.get('feature6',type=float)
        feature7 = request.form.get('feature7',type=float)
        feature8 = request.form.get('feature8',type=float)


        #
        input_data = pd.DataFrame([
            [
                feature1
            ]
        ], columns=[
            'pop'
        ])

        # 
        prediction = model.predict(input_data)
        # result = pd.DataFrame({'school': np.round(prediction[:,0]), 'academy': np.round(prediction[:,1]), 'transport': np.round(prediction[:,2]), 'APT': np.round(prediction[:,3]), 'receipts': np.round(prediction[:,4]), 'company': np.round(prediction[:,5]), 'hospital': np.round(prediction[:,6])})
        school = np.round(prediction[:,0])
        academy = np.round(prediction[:,1])
        transport = np.round(prediction[:,2])
        APT = np.round(prediction[:,3])
        receipts = np.round(prediction[:,4])
        company = np.round(prediction[:,5])
        hospital = np.round(prediction[:,6])

        # 'school', 'academy', 'transport', 'APT', 'receipts', 'company', 'hospital'
         
        return render_template('print_predict.html', school=school,
                                                     academy=academy,
                                                     transport=transport,
                                                     APT=APT,
                                                     receipts=receipts,
                                                     company=company,
                                                     hospital=hospital,
                                                     feature1=feature1,
                                                     feature2=feature2,
                                                     feature3=feature3,
                                                     feature4=feature4,
                                                     feature5=feature5,
                                                     feature6=feature6,
                                                     feature7=feature7,
                                                     feature8=feature8)
    

@app.route('/2_motivation')
def motivation():
    return render_template("motivation.html")

@app.route('/2_introduce')
def introduce():
    return render_template("introduce.html")

@app.route('/2_schedule')
def schedule():
    return render_template("schedule.html")

@app.route('/FQA', methods=['GET', 'POST'])
def FQA():
    try:
            print('select Started =====')
            mysql = getConnection()
            cur = mysql.cursor()
            cur.execute('select * from items')  
            rows = cur.fetchall() 
            desc = cur.description
            mysql.commit()
            cur.close()
            mysql.close()
            return render_template("FQA.html", rows=rows, desc=desc)
    except Exception as e:
        print(e)
        return render_template("fail.html")






#class PersonForm(FlaskForm): # 플라스크폼을 상속 해서 폼구조만 만든거
#     pop = StringField('pop',
#                validators=[InputRequired('입력필요!'),
#                Length(min=4, max=6, message='4 to 6자로 입력')])
#     submit = SubmitField('Submit')

# lr = joblib.load("my_model.pkl")
# model_columns = joblib.load("model_columns.pkl")


# def modelpredict(form):
#     if lr:
#         try:
#             form = PersonForm()
#             data = {'pop':form.pop.data}
#             query = pd.DataFrame(data, index=[0]) #, columns=['Age', 'Sex', 'Embarked']))

#             prediction = list(lr.predict(query))
#             print(prediction)
#             return prediction
#         except:
#               return {'trace': traceback.format_exc()}
#     else:
#         print ('모델먼저로딩해주세요')

# @app.route('/predict', methods=['GET','POST'])
# def form():
#     form = PersonForm()
#     prediction_result = None

#     if form.validate_on_submit():
#         prediction_result = modelpredict(form)
#         return prediction_result
    
#     return render_template('predict.html', form=form, prediction_result=prediction_result)

# 'school', 'academy', 'transport', 'APT', 'receipts', 'company', 'hospital'




# import os

# 현재 스크립트 파일의 경로를 기준으로 상대 경로 지정
# model_path = os.path.join(os.path.dirname(__file__), 'pop_density.pkl')
# model_path = os.path.join(os.path.dirname(__file__), 'dao', 'pop_density.pkl')
# os.chdir(os.path.dirname(__file__))  # 스크립트 파일이 있는 디렉토리로 변경
# model_path = 'dao/pop_density.pkl'

# model_path = os.path.join(os.path.dirname(__file__), 'pop.pkl')
#model = joblib.load('pop.pkl')



#@app.route('/predict', methods=['GET', 'POST'])
#def predict():
##    if request.method == 'POST':
#        feature_values = request.form.getlist('feature')
#        features = [float(value) for value in feature_values]
#        prediction = model.predict([features])[0]
#        return render_template('result.html', prediction=prediction)

#    return render_template('form.html')


if __name__=='__main__':
   app.run(debug=True)
