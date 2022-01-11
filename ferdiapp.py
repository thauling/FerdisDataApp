import streamlit as st
import pandas as pd
import numpy as np
#import altair as alt
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from urllib.error import URLError

df = pd.read_csv(r".\testbatch_utf8.csv", sep=';', decimal=",").dropna()


df = df.loc[3:]
print(df.head(10))
print(f"df shape: {df.shape}")
dft = df.T
print(f"df index: {dft.index}")
print(df.dtypes)

#convert all obj to float except for PDatTime
for col in list(dft.index[1:]):
  df[col] = df[col].str.replace(',', '.', regex=False)
  df[col] = df[col].astype(float)

# convert the 'PDatTime' column to datetime format
df['PDatTime'] = df['PDatTime'].astype('datetime64[ns]')

# Check the format of 'Date' column
print(df.info())
print(df.describe())

params = st.multiselect(
        "Choose parameters", list(dft.index[1:])) #, ["Age", "Subst", "pH"]

df = df.loc[3:]
dfselect = df[params]
#st.line_chart(df[params])

    
st.dataframe(df.head(10))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])
    
#st.line_chart(chart_data)   
#st.dataframe(chart_data.head())  


if len(params) == 2:
    
    y1 = params[0]
    y2 = params[1]
#print(y1, y2)

# Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
# Add traces
    fig.add_trace(
        go.Scatter(x=df["PDatTime"], y=df[y1], name=y1),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["PDatTime"], y=df[y2], name=y2),
        secondary_y=True,
    )
# Add figure title
    fig.update_layout(
        title_text="Ferdi's Double Y Analysis"
    )
# Set x-axis title
    fig.update_xaxes(title_text="DateTime")
# Set y-axes titles
    fig.update_yaxes(title_text=f"<b>primary</b> {y1}", secondary_y=False)
    fig.update_yaxes(title_text=f"<b>secondary</b> {y2}", secondary_y=True)

    #display chart
    st.plotly_chart(fig)
    st.table(dfselect.head(5))
else:
    #st.write("<b>Choose TWO parameters</b>")
    message = '<p style="font-family:Impact,sans-serif; color:#696969; font-size: 42px;">Choose TWO parameters</p>'
    st.markdown(message, unsafe_allow_html=True)