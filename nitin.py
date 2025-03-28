import streamlit as st 
import pandas as pd
pd=pd.read_csv("Ice_cream selling data.csv")
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
df=PolynomialFeatures(degree=2)
import joblib
model=joblib.load("OOO.joblib")
st.title("Ice Cream selling according to Temperature(°C) ") 
a=st.sidebar.number_input("Enter Ice Temperature (°C)",max_value=50,value=None)
btn=st.sidebar.button("Enter")
st.scatter_chart(pd,x="Temperature (°C)",y="Ice Cream Sales (units)")
st.subheader("The output show below here :")

if btn:
    if a is not None:
    
     b=np.array(a).reshape(-1,1)
     c=df.fit_transform(b)
     e=model.predict(c)
    
    
    
     st.success(f"The sold ice cream units is : {e[0]:.2f} ")
    else:
        st.error("Please enter all required values before predicting.")


