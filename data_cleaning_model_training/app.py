import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn

pipe=pickle.load(open('random_forest_model.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
# st.text(df)
st.title("Laptop Price Predictor")

#brand
company=st.selectbox("Brand",df["Company"].unique())

#type of laptop
type1=st.selectbox("Type",df["TypeName"].unique())

#Ram
ram=st.selectbox('Ram in GB',[2,4,6,8,12,16,24,32])

#weight
weight=st.number_input("Weight")

#TouchScreen
touchscreen=st.selectbox("Touchscreen",["No","Yes"])

#IPS display
ips=st.selectbox("IPS",["No","Yes"])

#screensize
screensize=st.number_input("Screen Size")

#resolution
resolution=st.selectbox("Screen Resolution",[
    '1366x768', '1600x900', '1920x1080', '1920x1200', '2160x1440',
    '2256x1504', '2496x1664', '2560x1440', '2560x1600', '2560x1664',
    '2880x1800', '2880x1864', '2880x1920', '3000x2000', '3024x1964',
    '3200x2000', '3456x2234', '3840x2160', '3840x2400'
])

#Cpu
cpu=st.selectbox("CPU Brand",df["Cpu_brand"].unique())

hdd=st.selectbox("HDD(in GB)",[0,128,256,512,1024,2048])

ssd=st.selectbox("SSD(in GB)",[0,128,256,512,1024])

gpu=st.selectbox("GPU Brand",df["Gpu_Brand"].unique())

os=st.selectbox("OS",df["os"].unique())

if st.button("Predict"):
    #query
    ppi=None
    if touchscreen=="Yes":
        touchscreen=1
    else:
        touchscreen=0
    if ips=="Yes":
        ips=1
    else:
        ips=0
    x_res=int(resolution.split("x")[0])
    y_res=int(resolution.split("x")[1])
    ppi=(((x_res**2)+(y_res**2))**0.5)/screensize
    query=pd.DataFrame([[company,type1,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os]],columns=["Company","TypeName","Ram","Weight","Touchscreen","IPS","ppi","Cpu_brand","HDD","SSD","Gpu_Brand","os"])
    X_enc = pipe.named_steps['step1'].transform(query)
    X_clean = np.ascontiguousarray(X_enc, dtype=np.float32)

    # Predict directly from the estimator
    pred = pipe.named_steps['step2'].predict(X_clean)
    st.title("The price for a laptop of this configuration will be around: {}rs".format(round(np.exp(pred[0]))))