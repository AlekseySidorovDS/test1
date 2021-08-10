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
        parser.add_argument('prev_cls2cur_opn_dd')
        parser.add_argument('prev_max_dpd')
        parser.add_argument('prev_cred_max_dpd_lp')
        parser.add_argument('gender')
        parser.add_argument('working_industry')
        parser.add_argument('ddong_hist_found')
        parser.add_argument('prev_cic_loans')
        parser.add_argument('age')
        parser.add_argument('JS_VAR_132')
        parser.add_argument('JS_VAR_101')
        parser.add_argument('gen_mar')
        parser.add_argument('cur_cic_loans')
        parser.add_argument('JS_VAR_159')
        parser.add_argument('antifraud_score')
        
        args=parser.parse_args()
        
        X=[args["prev_cls2cur_opn_dd"], args["prev_max_dpd"], args["prev_cred_max_dpd_lp"], args["gender"], args["working_industry"], args["ddong_hist_found"], args["prev_cic_loans"], args["age"], args["JS_VAR_132"], args["JS_VAR_101"], args["gen_mar"], args["cur_cic_loans"], args["JS_VAR_159"], args["antifraud_score"]]
        data = np.array(X)
        data = np.reshape(data,(1, -1))
        
        #DATEDIFF (day,prev_cred_cls,cur_app_date) as prev_cls2cur_opn_dd
        #prev_dpd.bad as prev_max_dpd
        #prev_dpd_last_paym.days_overdue as prev_cred_max_dpd_lp
        #case when pd_sex = 'MALE' then 1 else 0 end as gender
        #  case when ed_working_industry = 'MANUFACTURING' then 3 when ed_working_industry = 'SERVICES' then 2
	   #when ed_working_industry = 'TRADING' then 1 when ed_working_industry = 'TRANSPORTATION' then 5
	   #when ed_working_industry = 'CONTRUCTIONS' then 4 else 0 end as working_industry
        #ddong.history_found as ddong_hist_found
        #prev_cic.cnt_loans as prev_cic_loans
        #DATEDIFF (YEAR,pd_birth_date,cur_app_date) as age
        #juicy_full.JS_VAR_132
        #juicy_full.JS_VAR_101
        #case when (pd_sex = 'FEMALE' and pd_marital_status = 'MARRIED') or (pd_sex = 'FEMALE' and pd_marital_status = 'DIVORCED') then 2
	   #when (pd_sex = 'FEMALE' and pd_marital_status = 'SINGLE') then 3
	   #when (pd_sex = 'MALE' and pd_marital_status = 'MARRIED') then 1
	   #else 0 end as gen_mar
        #cur_cic_loans
        #juicy_full.JS_VAR_159
        #juicy_full.antifraud_score
        
        
        columns = ['prev_cls2cur_opn_dd', 'prev_max_dpd', 'prev_cred_max_dpd_lp', 'gender', 'working_industry', 'ddong_hist_found', 'prev_cic_loans', 'age', 'JS_VAR_132', 'JS_VAR_101', 'gen_mar', 'cur_cic_loans', 'JS_VAR_159', 'antifraud_score']
        df = pd.DataFrame(data = data,  columns = columns) 
        df = df.astype(float)
        
        df["bin_prev_cls2cur_opn_dd"] = ""
        df.loc[df["prev_cls2cur_opn_dd"] <= 4, "bin_prev_cls2cur_opn_dd"] = "(1.0, 4.0]"
        df.loc[df["prev_cls2cur_opn_dd"] <= 1, "bin_prev_cls2cur_opn_dd"] = "(0.0, 1.0]"
        df.loc[df["prev_cls2cur_opn_dd"] <= 0, "bin_prev_cls2cur_opn_dd"] = "(-inf, 0.0]"
        df.loc[df["prev_cls2cur_opn_dd"] > 4, "bin_prev_cls2cur_opn_dd"] = "(4.0, inf]"
        
        df["woe_prev_cls2cur_opn_dd"] = ""
        df.loc[df["bin_prev_cls2cur_opn_dd"] == "(-inf, 0.0]", "woe_prev_cls2cur_opn_dd"] = -0.271295
        df.loc[df["bin_prev_cls2cur_opn_dd"] == "(0.0, 1.0]", "woe_prev_cls2cur_opn_dd"] = 0.184588
        df.loc[df["bin_prev_cls2cur_opn_dd"] == "(1.0, 4.0]", "woe_prev_cls2cur_opn_dd"] = 0.184588
        df.loc[df["bin_prev_cls2cur_opn_dd"] == "(4.0, inf]", "woe_prev_cls2cur_opn_dd"] = 0.422300
#
        df["bin_prev_max_dpd"] = ""
        df.loc[df["prev_max_dpd"] <= 0, "bin_prev_max_dpd"] = "(-inf, 0.0]"
        df.loc[df["prev_max_dpd"] > 0, "bin_prev_max_dpd"] = "(0.0, 7.0]"
        
        df["woe_prev_max_dpd"] = ""
        df.loc[df["bin_prev_max_dpd"] == "(-inf, 0.0]", "woe_prev_max_dpd"] =  0.076279 
        df.loc[df["bin_prev_max_dpd"] == "(0.0, 7.0]", "woe_prev_max_dpd"] = -0.862259
        
        
        df["bin_prev_cred_max_dpd_lp"] = ""
        df.loc[df["prev_cred_max_dpd_lp"] !=0, "bin_prev_cred_max_dpd_lp"] = "nan"
        df.loc[df["prev_cred_max_dpd_lp"] <= 0, "bin_prev_cred_max_dpd_lp"] = "(-inf, 0.0]"
        df.loc[df["prev_cred_max_dpd_lp"] > 0, "bin_prev_cred_max_dpd_lp"] = "(0.0, 7.0]"        
        
        df["woe_prev_cred_max_dpd_lp"] = ""
        df.loc[df["bin_prev_cred_max_dpd_lp"] == "(-inf, 0.0]", "woe_prev_cred_max_dpd_lp"] =  0.090942 
        df.loc[df["bin_prev_cred_max_dpd_lp"] == "(0.0, 7.0]", "woe_prev_cred_max_dpd_lp"] = -1.002940
        df.loc[df["bin_prev_cred_max_dpd_lp"] == "nan", "woe_prev_cred_max_dpd_lp"] = 0.090942
        
        df["bin_working_industry"] = ""
        df.loc[df["working_industry"] <= 1, "bin_working_industry"] = "(-inf, 1.0]"
        df.loc[df["working_industry"] > 1, "bin_working_industry"] = "(1.0, 5.0]"   
        
        df["woe_working_industry"] = ""
        df.loc[df["bin_working_industry"] == "(-inf, 1.0]", "woe_working_industry"] = 0.249063
        df.loc[df["bin_working_industry"] == "(1.0, 5.0]", "woe_working_industry"] = -0.125085
#
        df["bin_age"] = ""
        df.loc[df["age"] <= 38, "bin_age"] = "(29.0, 38.0]"
        df.loc[df["age"] <= 29, "bin_age"] = "(22.0, 29.0]"
        df.loc[df["age"] <= 22, "bin_age"] = "(-inf, 22.0]"
        df.loc[df["age"] > 38, "bin_age"] = "(38.0, inf]"
        
        df["woe_age"] = ""
        df.loc[df["bin_age"] == "(-inf, 22.0]", "woe_age"] = -0.164949
        df.loc[df["bin_age"] == "(22.0, 29.0]", "woe_age"] = -0.164949
        df.loc[df["bin_age"] == "(29.0, 38.0]", "woe_age"] = 0.012457
        df.loc[df["bin_age"] == "(38.0, inf]", "woe_age"] = 0.216465 
#
        df["bin_gen_mar"] = ""
        df.loc[df["gen_mar"] <= 2, "bin_gen_mar"] = "(0.0, 2.0]"
        df.loc[df["gen_mar"] <= 0, "bin_gen_mar"] = "(-inf, 0.0]"
        df.loc[df["gen_mar"] > 2, "bin_gen_mar"] = "(2.0, inf]"
        
        df["woe_gen_mar"] = ""
        df.loc[df["bin_gen_mar"] == "(-inf, 0.0]", "woe_gen_mar"] = -0.200797
        df.loc[df["bin_gen_mar"] == "(0.0, 2.0]", "woe_gen_mar"] = 0.057451
        df.loc[df["bin_gen_mar"] == "(2.0, inf]", "woe_gen_mar"] = 0.167077
#
        df["bin_gender"] = ""
        df.loc[df["gender"] > 0, "bin_gender"] = "(0.0, 1.0]"
        df.loc[df["gender"] <= 0, "bin_gender"] = "(-inf, 0.0]"
        
        df["woe_gender"] = ""
        df.loc[df["bin_gender"] == "(-inf, 0.0]", "woe_gender"] = 0.190494
        df.loc[df["bin_gender"] == "(0.0, 1.0]", "woe_gender"] = -0.157285
        
        df["bin_ddong_hist_found"] = ""
        df.loc[df["ddong_hist_found"] > 0, "bin_ddong_hist_found"] = "(0.0, 1.0]"
        df.loc[df["ddong_hist_found"] <= 0, "bin_ddong_hist_found"] = "(-inf, 0.0]"
        
        df["woe_ddong_hist_found"] = ""
        df.loc[df["bin_ddong_hist_found"] == "(-inf, 0.0]", "woe_ddong_hist_found"] =  0.085727 
        df.loc[df["bin_ddong_hist_found"] == "(0.0, 1.0]", "woe_ddong_hist_found"] = -0.260174
        
        df["bin_prev_cic_loans"] = ""
        df.loc[df["prev_cic_loans"] != 0, "bin_prev_cic_loans"] = "nan"
        df.loc[df["prev_cic_loans"] <= 2, "bin_prev_cic_loans"] = "(0.0, 2.0]"
        df.loc[df["prev_cic_loans"] <= 0, "bin_prev_cic_loans"] = "(-inf, 0.0]"
        df.loc[df["prev_cic_loans"] > 2, "bin_prev_cic_loans"] = "(2.0, inf]"
        
        df["woe_prev_cic_loans"] = ""
        df.loc[df["bin_prev_cic_loans"] == "(-inf, 0.0]", "woe_prev_cic_loans"] = -0.301815
        df.loc[df["bin_prev_cic_loans"] == "(0.0, 2.0]", "woe_prev_cic_loans"] = -0.029920
        df.loc[df["bin_prev_cic_loans"] == "(2.0, inf]", "woe_prev_cic_loans"] = 0.065695
        df.loc[df["bin_prev_cic_loans"] == "nan", "woe_prev_cic_loans"] = 0.290420
        
        df["bin_JS_VAR_132"] = ""
        df.loc[df["JS_VAR_132"] != 0, "bin_JS_VAR_132"] = "nan"
        df.loc[df["JS_VAR_132"] <= 6377, "bin_JS_VAR_132"] = "(3618.0, 6377.0]"
        df.loc[df["JS_VAR_132"] <= 3618, "bin_JS_VAR_132"] = "(142.0, 3618.0]"
        df.loc[df["JS_VAR_132"] <= 142, "bin_JS_VAR_132"] = "(-inf, 142.0]"
        df.loc[df["JS_VAR_132"] > 6377, "bin_JS_VAR_132"] = "(6377.0, 48800.0]"
        
        df["woe_JS_VAR_132"] = ""
        df.loc[df["bin_JS_VAR_132"] == "(-inf, 142.0]", "woe_JS_VAR_132"] = 1.625942
        df.loc[df["bin_JS_VAR_132"] == "(142.0, 3618.0]", "woe_JS_VAR_132"] = 0.137369
        df.loc[df["bin_JS_VAR_132"] == "(3618.0, 6377.0]", "woe_JS_VAR_132"] =   -0.266288
        df.loc[df["bin_JS_VAR_132"] == "(6377.0, 48800.0]", "woe_JS_VAR_132"] =   -0.704814
        df.loc[df["bin_JS_VAR_132"] == "nan", "woe_JS_VAR_132"] = -0.00630
        
        df["bin_JS_VAR_101"] = ""
        df.loc[df["JS_VAR_101"] != 0, "bin_JS_VAR_101"] = "nan"
        df.loc[df["JS_VAR_101"] <= 3, "bin_JS_VAR_101"] = "(1.0, 3.0]"
        df.loc[df["JS_VAR_101"] <= 1, "bin_JS_VAR_101"] = "(-inf, 1.0]"
        df.loc[df["JS_VAR_101"] > 3, "bin_JS_VAR_101"] = "(3.0, inf]"
        
        df["woe_JS_VAR_101"] = ""
        df.loc[df["bin_JS_VAR_101"] == "(-inf, 1.0]", "woe_JS_VAR_101"] =  -0.103094
        df.loc[df["bin_JS_VAR_101"] == "(1.0, 3.0]", "woe_JS_VAR_101"] = -0.084139  
        df.loc[df["bin_JS_VAR_101"] == "(3.0, inf]", "woe_JS_VAR_101"] = 0.084764  
        df.loc[df["bin_JS_VAR_101"] == "nan", "woe_JS_VAR_101"] = -0.014698
        
        df["bin_cur_cic_loans"] = ""
        df.loc[df["cur_cic_loans"] != 0, "bin_cur_cic_loans"] = "nan"
        df.loc[df["cur_cic_loans"] <= 1, "bin_cur_cic_loans"] = "(0.0, 1.0]"
        df.loc[df["cur_cic_loans"] <= 0, "bin_cur_cic_loans"] = "(-inf, 0.0]"
        df.loc[df["cur_cic_loans"] > 1, "bin_cur_cic_loans"] = "(1.0, inf]"
        
        df["woe_cur_cic_loans"] = ""
        df.loc[df["bin_cur_cic_loans"] == "(-inf, 0.0]", "woe_cur_cic_loans"] = -0.343684
        df.loc[df["bin_cur_cic_loans"] == "(0.0, 1.0]", "woe_cur_cic_loans"] = -0.032530
        df.loc[df["bin_cur_cic_loans"] == "(1.0, inf]", "woe_cur_cic_loans"] = 0.031839
        df.loc[df["bin_cur_cic_loans"] == "nan", "woe_cur_cic_loans"] = 0.172439
        
        df["bin_JS_VAR_159"] = ""
        df.loc[df["JS_VAR_159"] != 0, "bin_JS_VAR_159"] = "nan"
        df.loc[df["JS_VAR_159"] <= 2400, "bin_JS_VAR_159"] = "(480.0, 2400.0]"
        df.loc[df["JS_VAR_159"] <= 480, "bin_JS_VAR_159"] = "(-inf, 480.0]"
        df.loc[df["JS_VAR_159"] > 2400, "bin_JS_VAR_159"] = "(2400.0, inf]"
        
        df["woe_JS_VAR_159"] = ""
        df.loc[df["bin_JS_VAR_159"] == "(-inf, 480.0]", "woe_JS_VAR_159"] = 0.033357
        df.loc[df["bin_JS_VAR_159"] == "(480.0, 2400.0]", "woe_JS_VAR_159"] = -0.036365
        df.loc[df["bin_JS_VAR_159"] == "(2400.0, inf]", "woe_JS_VAR_159"] =   0.524485
        df.loc[df["bin_JS_VAR_159"] == "nan", "woe_JS_VAR_159"] = -0.010646
        
        df["bin_antifraud_score"] = ""
        df.loc[df["antifraud_score"] != 0, "bin_antifraud_score"] = "nan"
        df.loc[df["antifraud_score"] <= 0.49, "bin_antifraud_score"] = "(-inf, 0.49]"
        df.loc[df["antifraud_score"] > 0.49, "bin_antifraud_score"] = "(0.49, 0.98]"
        
        df["woe_antifraud_score"] = ""
        df.loc[df["bin_antifraud_score"] == "(-inf, 0.49]", "woe_antifraud_score"] =  0.081754
        df.loc[df["bin_antifraud_score"] == "(0.49, 0.98]", "woe_antifraud_score"] = -0.345868
        df.loc[df["bin_antifraud_score"] == "nan", "woe_antifraud_score"] = -0.014698
        
        cols = [c for c in df.columns.values if c.startswith('woe_')]
        woe_df = df[cols]
                
         #index = [1],                 
        result = XG_MODEL_R1_START_V1.predict_proba(woe_df)[:,1]
        result = result.tolist()
        return result, 200
        #return df, 200
        
api.add_resource(Predict_proba,"/XG_MODEL_R1_START_V1/")

if __name__ == "__main__":
   app.run()
