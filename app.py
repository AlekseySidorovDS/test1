from flask import Flask
from flask_restful import Api, Resource, reqparse
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd 
filename = 'XG_MODEL_R1_START_V1.sav'
XG_MODEL_R1_START_V1 = pickle.load(open(filename, 'rb'))
app=Flask(__name__)
api=Api(app)

class Predict_proba (Resource):
    
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('woe_prev_cls2cur_opn_dd')
        parser.add_argument('woe_prev_max_dpd')
        parser.add_argument('woe_prev_cred_max_dpd_lp')
        parser.add_argument('woe_gender')
        parser.add_argument('woe_working_industry')
        parser.add_argument('woe_ddong_hist_found')
        parser.add_argument('woe_prev_cic_loans')
        parser.add_argument('woe_age')
        parser.add_argument('woe_JS_VAR_132')
        parser.add_argument('woe_JS_VAR_101')
        parser.add_argument('woe_gen_mar')
        parser.add_argument('woe_cur_cic_loans')
        parser.add_argument('woe_JS_VAR_159')
        parser.add_argument('woe_antifraud_score')
        
        args=parser.parse_args()
        X=[args["woe_prev_cls2cur_opn_dd"], args["woe_prev_max_dpd"], args["woe_prev_cred_max_dpd_lp"], args["woe_gender"], args["woe_working_industry"], args["woe_ddong_hist_found"], args["woe_prev_cic_loans"], args["woe_age"], args["woe_JS_VAR_132"], args["woe_JS_VAR_101"], args["woe_gen_mar"], args["woe_cur_cic_loans"], args["woe_JS_VAR_159"], args["woe_antifraud_score"]]
        test = np.array(X)
        test = np.reshape(test,(1, -1))
        
        columns = ['woe_prev_cls2cur_opn_dd', 'woe_prev_max_dpd', 'woe_prev_cred_max_dpd_lp', 'woe_gender', 'woe_working_industry', 'woe_ddong_hist_found', 'woe_prev_cic_loans', 'woe_age', 'woe_JS_VAR_132', 'woe_JS_VAR_101', 'woe_gen_mar', 'woe_cur_cic_loans', 'woe_JS_VAR_159', 'woe_antifraud_score']
        df = pd.DataFrame(data = test, index = [1], columns = columns)                   
                          
      

        return XG_MODEL_R1_START_V1.predict_proba(df)[:,1], 200
        #return test, 200
        
api.add_resource(Predict_proba,"/XG_MODEL_R1_START_V1/")

if __name__ == "__main__":
   app.run()
