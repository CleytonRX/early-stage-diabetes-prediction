import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from sklearn.preprocessing import StandardScaler


def load_images(image_name):
    img = Image.open(image_name)
    return st.image(img, width=800)

feature_names_best = ['Polyuria', 'Polydipsia','Age', 'Gender','partial paresis','sudden weight loss',
                      'Irritability', 'delayed healing','Alopecia','Itching']


gender_dict = {"Masculino":1,"Feminino":0}
feature_dict = {"Yes":1,"No":0}

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 
              
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

import lime
import lime.lime_tabular


def main():


    st.title('CRX Analytics 🤖')
    st.subheader('Prevendo o Risco de Diabetes Mellitus tipo 2')
    
    menu = ["Home"]
    #submenu = [ "Plot" , "Prediction", "Analytics" ]
    
    choice = st.sidebar.selectbox ("Menu", menu)
    if choice == "Home":
        #st.subheader("Página Inicial")
        st.text( "Preencha o formulário abaixo e clique em enviar para saber se você pode ter diabetes.")
        #c_image = 'diab.png'
        #load_images(c_image)
                    
	                        
			
		
				
    #elif activity == "Prediction":
        st.subheader("Análise Preditiva")
        Age = st.number_input("Informe idade entre 7 e 80 anos",7,80) 
        Gender = st.radio("Gênero",tuple(gender_dict.keys()))
        Polyuria = st.radio("Você tem poliúria?",tuple(feature_dict.keys()))
        Polydipsia = st.radio("Você tem Polidipsia?",tuple(feature_dict.keys()))
        Sudden_weight_loss = st.radio("Você recentemente teve perda de peso repentina?",tuple(feature_dict.keys()))
        Weakness = st.radio("Você costuma ter fraqueza muscular?",tuple(feature_dict.keys()))
        Polyphagia = st.radio("Você experimentou recentemente fome excessiva ou aumento do apetite?",tuple(feature_dict.keys()))
        Genital_thrush = st.radio("Você tem candidíase genital?",tuple(feature_dict.keys()))
        Visual_blurring = st.radio("Você experimentou recentemente um desfoque visual (vista embaçada)?",tuple(feature_dict.keys()))
        Itching = st.radio("Você experimentou recentemente coceira inexplicável?",tuple(feature_dict.keys()))
        Irritability = st.radio("Você experimentou recentemente irritabilidade excessiva?",tuple(feature_dict.keys()))
        Delayed_healing = st.radio("Você experimentou recentemente um maior tempo para cicatrização de feridas?",tuple(feature_dict.keys()))
        Partial_paresis = st.radio("Você experimentou recentemente paresia parcial? (restrição/diminuição do movimento)",tuple(feature_dict.keys()))
        Muscle_stiffness = st.radio("Você recentemente experimentou rigidez muscular?",tuple(feature_dict.keys()))
        Alopecia = st.radio("Você tem alopecia? (queda de cabelo irregular)",tuple(feature_dict.keys()))
        Obesity = st.radio("Você tem obesidade com base no seu IMC?",tuple(feature_dict.keys()))                      
        feature_list = [Age,
                        get_value(Gender,gender_dict),get_fvalue(Polyuria),
                        get_fvalue(Polydipsia),get_fvalue(Sudden_weight_loss),
                        get_fvalue(Weakness),get_fvalue(Polyphagia),
                        get_fvalue(Genital_thrush),get_fvalue(Visual_blurring),
                        get_fvalue(Itching),get_fvalue(Irritability),
                        get_fvalue(Delayed_healing),get_fvalue(Partial_paresis),
                        get_fvalue(Muscle_stiffness),get_fvalue(Alopecia),
                        get_fvalue(Obesity)]

        st.write(len(feature_list))			
        pretty_result = {"Age":Age,"Gender":Gender,"Polyuria":Polyuria,"Polydipsia":Polydipsia,
                         "Sudden_weight_loss":Sudden_weight_loss,"Weakness":Weakness,
                         "Polyphagia":Polyphagia,"Genital_thrush":Genital_thrush,
                         "visual_blurring":Visual_blurring,"Itching":Itching,
                         "Irritability":Irritability,"Delayed_healing":Delayed_healing,
                         "Partial_paresis":Partial_paresis,"Muscle_stiffness":Muscle_stiffness,
                         "Alopecia":Alopecia,"Obesity":Obesity}

        st.json(pretty_result)
        
        single_sample = np.array(feature_list).reshape(1,-1)
        
        if st.button("Ver Resultado"):

                    #model_choice == "RF":
                    loaded_model = load_model("models/random_forest.joblib")
                    prediction = loaded_model.predict(single_sample)
                    pred_prob = loaded_model.predict_proba(single_sample)


                    if prediction == 0:
                            st.warning("Não há risco de ser Diabético")
                            pred_probability_score = {"Probabilidade de não ser Diabético":pred_prob[0][0]*100,"Probabilidade de ser Diabético":pred_prob[0][1]*100}
                            #st.subheader("Prediction Probability Score using {}".format(model_choice))
                            st.json(pred_probability_score)
                            
                    else:
                            st.success("Há risco de ser Diabético")
                            pred_probability_score = {"Probabilidade de não ser Diabético":pred_prob[0][0]*100,"Probabilidade de ser Diabético":pred_prob[0][1]*100}
                            #st.subheader("Prediction Probability Score using {}".format(model_choice))
                            st.json(pred_probability_score)                                  
                            


			
					
if __name__ == '__main__' :
    main()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
