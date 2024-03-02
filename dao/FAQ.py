import json 
import pymysql
# business logic -> controller -> html(client:web browser)
# connection객체 
def getConnection():
    return pymysql.connect(host='localhost', port=3306, user='root', passwd='moon154848!@',
                     db='sungjuk', charset='utf8', autocommit=True)

