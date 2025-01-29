import streamlit as st
import pandas as pd 
import joblib 
import numpy as np
import time as time
    
scaler= joblib.load('scaler.pkl')

model = joblib.load('model.pkl')

st.set_page_config(page_title="hyper_test", page_icon="🩺")
st.title("**THIS IS HYPERTENSION PROJECT**")
st.subheader("**inputs**")

col1,col2=st.columns(2)



age=col1.number_input("**Age**(10-100)",min_value=10,max_value=100,value="min")



sex=col2.selectbox("**Select your sex**",["Male","Female"])
if sex == "Male":
    sex = 1
else:
    sex = 0



cp=col1.selectbox("**Select your chest pain type**",["asymptomatic ","typical angina ","atypical angina","non-anginal pain"])
if cp == "asymptomatic ":
    cp = 0
elif cp == "typical angina ":
    cp = 1
elif cp == "atypical angina":
    cp=2
elif cp == "non-anginal pain":
    cp=3



trestbps=col2.number_input("**Resting blood pressure** (in mm Hg)",min_value=80,max_value=200,value="min")


chol=col1.number_input(" **Serum cholestoral** (in mg/dl)",min_value=100,max_value=600,value="min")


fbs=col2.selectbox("**if the patient's fasting blood sugar > 120 mg/dl**",["yes","no"])
if fbs == "yes":
    fbs = 1
else:
    fbs = 0


restecg=col1.selectbox("**Resting ECG results**",["Normal","ST-T Wave Abnormality","Severe ST-T Wave Abnormality"])
if restecg == "Normal":
    restecg = 0
elif restecg == "ST-T Wave Abnormality":
    restecg = 1
elif restecg == "Severe ST-T Wave Abnormality":
    restecg = 2


thalach=col2.number_input(" **Maximum heart rate achieved**",min_value=60,max_value=200,value="min")



exang=col1.selectbox("**Exercise induced angina**",["yes","no"])
if exang == "yes":
    exang = 1
else:
    exang = 0



oldpeak=col2.number_input(" **ST depression induced by exercise relative to rest**",min_value=0.0,max_value=7.0,value="min",step=0.1)


slope=col1.selectbox("**Slope of the peak exercise ST segment**",["upsloping","flat","downsloping"]) 
if slope == "upsloping":
    slope = 0
elif slope == "flat":
    slope = 1
elif slope == "downsloping":
    slope = 2


ca=col2.number_input(" **Number of major vessels colored by flourosopy**(0-4)",min_value=0,max_value=4,value="min",step=1)


thal=col1.selectbox("**myocardial perfusion scan**",["normal","fixed defect","reversable defect"])
if thal == "normal":
    thal = 1
elif thal == "fixed defect":
    thal = 2
elif thal == "reversable defect":
    thal = 3






inputs=[age, sex, cp, trestbps, chol, fbs, restecg,thalach, exang, oldpeak, slope, ca, thal]
inputs=np.array(inputs).reshape(1,-1)
inputs=scaler.transform(inputs)
result=model.predict(inputs)

neg_perc=model.predict_proba(inputs)[0][0]
pos_perc=model.predict_proba(inputs)[0][1]

perc=''
if neg_perc>pos_perc:
    perc= str(neg_perc*100)+'%'
else:
    perc= str(pos_perc*100)+'%'


text_result=''
if result ==1:
    text_result="The patient has a hypertension"
else:
    text_result="The patient does not have a hypertension"




start_text = f"""
This is just an expectation
 from a program and not sure 
 please check with your doctor
but this program expects that
**{text_result}** with **{perc}** We hope that you are always
in good health \n\n

"""
st.write("-"*50)
end_text="""**WARNING**\n


It’s essential to stay calm and take immediate steps to manage your condition until you can see a doctor. Here are some things you can do:\n

Relax: Try to stay calm and take deep breaths. Stress can increase blood pressure.\n


Avoid Stimulants: Stay away from caffeine, nicotine, and alcohol, as these can raise your blood pressure.\n

Stay Hydrated: Drink plenty of water to help maintain healthy blood pressure levels.\n

Monitor Your Blood Pressure: If you have a blood pressure monitor, keep track of your readings.\n

Eat Healthily: Focus on fruits, vegetables, and low-sodium foods. Avoid processed foods and excess salt.\n

Rest: Take breaks and avoid heavy physical activity.\n

Seek Medical Attention if Necessary: If you experience symptoms like chest pain, severe headache, dizziness, or shortness of breath, seek immediate medical help.

"""

def stream_data():
    for word in start_text.split(" "):
        yield word + " "
        time.sleep(0.02)

    if result==1:
        for word in end_text.split(" "):
            yield word + " "
            time.sleep(0.02)


if st.button("**PREDICT**"):
    st.write_stream(stream_data)
