import streamlit as st
import pandas as pd
from prophet import Prophet

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pickle
import numpy as np
import plotly.graph_objects as go

from utils.load import load_image,load_toy_dataset
from utils.models import prophet_modeling,plot_performance_metrics, load_data

#from lib.module import inputs
#from lib.inputs.params import (
#    input_holidays_params,
#    input_prior_scale_params, 
#    input_seasonality_params
#    )

# Info
with st.expander("Prophet model to build a time series forecasting model in a few clicks", expanded=False):
    st.write("Prophet model")
    
st.write("Time series forecasting")
st.sidebar.title("DataGush")
#st.sidebar.image(load_image("prophet_logo.PNG"), use_column_width=True)

##
st.sidebar.title("1. Data")
# Sidebar for dataset selection
with st.sidebar.expander("Dataset", expanded=True):
    use_default_dataset = st.checkbox("Select the Dataset", True)
    # List of options 
    choices = ["3DS 2.0", "CGK1", "CATS External", "CATS Internal", "Rule Engine"] 
    #choices = ["3DS 2.0"]
    # Single-select dropdown 
    AppName = st.selectbox("Select an option", choices, index=0) 
    default_sheet_name = str(AppName) + " Hourly Data"
    #st.write("Sheet name: ", default_sheet_name)
  
    if use_default_dataset:
        default_file_path = "/var/MLP/Shubham/ApplicationHourlyVolume.xlsx"
        # default_sheet_name = st.selectbox("Select a Sheet", pd.ExcelFile(default_file_path).sheet_names, help="Select a sheet from the default Excel file")
        df = load_data(default_file_path, default_sheet_name)
        #date_col = df.columns[0]
        #target_col = df.column[1]
        df = df.rename(columns={'Date': "ds", 'Volume': "y"})
        holidays_data_path = "/var/MLP/Shubham/holidays.xlsx"
        holidays_df1 = pd.read_excel(holidays_data_path)
      
        # Get excel data and format
        inputData = pd.ExcelFile("/var/MLP/Shubham/CapacityInput.xlsx")
        try:
            serviceCapacityData = pd.read_excel(inputData, AppName + " Services")
        except:
            #if capacity data not available set to 0 so forecasting can continue
            serviceCapacityData = {'Services': [AppName], 'No. of Nodes': [0], 'Capacity/Node': [0], 'Current Capacity TPS': [0], 'Percentage': [1]}
        applicationData = pd.read_excel(inputData, "Applications")

    else:
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"], help="Upload your Excel dataset", accept_multiple_files=False)
        if uploaded_file:
            sheet_name = st.selectbox("Select a sheet", pd.ExcelFile(uploaded_file).sheet_names, help="Select a sheet from the uploaded Excel file")
            df = load_data(uploaded_file, sheet_name)

            # Column names
            if df is not None:
                with st.sidebar.expander("Columns", expanded=True):
                    date_col = st.selectbox("Date column",sorted(df.columns))
                    target_col = st.selectbox( "Target column", sorted(set(df.columns) - {date_col}) )
                    df = df.rename(columns={date_col: "ds", target_col: "y"})
                    
            else:
                st.write("Please Upload an Excel file")
                st.stop()
if df is None:
    st.stop()


# st.write(df)
###********************************************************************************
st.sidebar.title("2. Modelling")
# Split data into training and testing sets
st.sidebar.header('Data Splitting')
# Get the first and last date
first_date = df['ds'].min()
last_date = df['ds'].max()
# Display the first and last date in the sidebar
st.sidebar.write(f"First Date: {first_date.strftime('%Y-%m-%d')}")
st.sidebar.write(f"Last Date: {last_date.strftime('%Y-%m-%d')}")
# Slider for selecting the percentage of data for training
split_percentage = st.sidebar.slider('Select the percentage for training data', 0, 100, 100, step=10)
# Calculate the split index based on the selected percentage
split_index = int(len(df) * split_percentage / 100)

# Split the data
train = df[:split_index]
test = df[split_index:]
# Display the split data
st.sidebar.write(f"Training Data Length: {train.shape[0]}")
st.subheader('Training Data')
st.write(train.head())
st.sidebar.write(f"Testing Data Length: {test.shape[0]}")
st.subheader('Testing Data')
st.write(test.head())
 
# Custom Holidays data
# Holidays
#with st.sidebar.expander("Holidays"):
#    params = input_holidays_params(params, readme, config)

with st.sidebar.expander("Custom Holidays"):
    holidays_data = st.file_uploader(label="Upload a excel file with holiday and dates", type=["xlsx", "xls"], help="holiday_data_upload", accept_multiple_files=False)
    if holidays_data is not None:
        holidays_df = pd.read_excel(holidays_data)
    else:
        holidays_df = None
#if AppName == '3DS 2.0':
#    filename_3ds = "/var/MLP/Shubham/DataGush_model.sav"
#    loaded_model = pickle.load(open(filename_3ds, 'rb')) 
#else:

# ypred = loaded_model.predict(data)



##********************** Forecast *******************************    
st.sidebar.title("3. Forecast")
#forecast_data = st.sidebar.checkbox("Launch Forecast", value=True)
if use_default_dataset:
    holidays_df = holidays_df1
    if holidays_data is not None:
        holidays_df = holidays_df
    #st.write('Run predefined model')
    #if loaded_model:
    #    model = loaded_model
    if AppName == '3DS 2.0':
        filename_3ds = "/var/MLP/Shubham/DataGush_model.sav"
        loaded_model = pickle.load(open(filename_3ds, 'rb')) 
        model = loaded_model
    else:
        # Perform modeling
        model = prophet_modeling(holidays_df)
        model.fit(train)
else: 
    #st.write('Run model finetuning')
    model = finetuned_model(holidays_df)
    # Fit the model
    model.fit(train)
#st.sidebar.header('Future prediction')
numMonths = st.sidebar.number_input("Number of Periods(Months)", min_value=0, value=0)
#frequency = st.sidebar.selectbox("Frequency", ["D", "H", "W", "M"], index=0)
# Make future dataframe for the period of the test set

firsttwo = df['ds'].iloc[:2]
timedif = (firsttwo[1] - firsttwo[0]).total_seconds()
if timedif <= 3600:
    #st.write('hourly data')
    future= model.make_future_dataframe(periods=730*numMonths,freq='H')
elif timedif <= 86400:
    future= model.make_future_dataframe(periods=30*numMonths,freq='D')
elif timedif >=2419200:
    future= model.make_future_dataframe(periods=numMonths,freq='M')
else:
    future= model.make_future_dataframe(periods=730*numMonths,freq='H')
#future = model.make_future_dataframe(periods=730*2,freq='H')
 
# rmse = round(root_mean_squared_error(data.y,forecast.yhat[:train.shape[0]]))
# st.write('RMSE: ',rmse)
show_plots = st.sidebar.checkbox("Launch Forecast", value=False)

