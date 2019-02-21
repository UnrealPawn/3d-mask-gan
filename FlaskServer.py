import importlib
gan=importlib.import_module("3dgan_Larry_5")

import json
from flask import Flask,url_for
import os
import datetime




app = Flask(__name__)

@app.route('/')
def hello_world():
    return "GAN Server Started"
begin = datetime.datetime.now()
result_generater = gan.ganGetOneResult(r"D:\RockGan\3DGANServer\models\Larry_5\Larry_5_1_15210.cptk")
next(result_generater)
end = datetime.datetime.now()
print(end-begin)
#test: http://127.0.0.1:5000/gan/22%2622%260%2642%2642%2662
@app.route('/gan/<int:minx>&<int:miny>&<int:minz>&<int:maxx>&<int:maxy>&<int:maxz>')
def RequestGANGen(minx,miny,minz,maxx,maxy,maxz):
    #result=gan.ganGetOneResult(minx,miny,minz,maxx,maxy,maxz,"./models/biasfr ee_810.cptk")
    #return json.dumps(result.astype(int).tolist())
    begin = datetime.datetime.now()
    minx = 0
    miny = 0
    minz = 0
    maxx = 64
    maxy = 64
    maxz = 20
    #gan.SetBoundState(minx, miny, minz, maxx, maxy, maxz)
    result_list=next(result_generater)
    result_list=result_list.ravel()
    back=""
    for b in result_list:
        if b == True:
            back=back+"1"
        else:
            back=back+"0"
    end = datetime.datetime.now()
    print(end-begin)
    return back


with app.test_request_context():
    print (url_for('RequestGANGen', minx='22',miny='22',minz='0',maxx='42',maxy='42',maxz='62'))

if __name__ == '__main__':
    #app.debug=True
    app.run(host='0.0.0.0')