import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import pickle
# import datetime
# Function to load data
@st.cache_data
def load_data(file, sheet_name=None):
    if sheet_name:
        return pd.read_excel(file, sheet_name=sheet_name)
    else:
        return pd.read_excel(file)


# Function to perform Prophet modeling
def finetuned_model(holidays=None):
    model = Prophet(
        holidays=holidays
    )
    return model
def prophet_modeling(holidays=None):
    best_params =  {'seasonality_prior_scale': 0.02009233002565047,
                    'interval_width': 0.90, 
                    'holidays_prior_scale': 8.697490026177835, 
                    'changepoint_range': 0.7166666666666667, 
                    'changepoint_prior_scale': 10.0}
     model = Prophet(
            changepoint_prior_scale = best_params['changepoint_prior_scale'],
            seasonality_prior_scale = best_params['seasonality_prior_scale'],
            holidays_prior_scale = best_params['holidays_prior_scale'],
            seasonality_mode = 'multiplicative',
     )
    model.add_seasonality(name = 'weekly', period = 7, mode = 'multiplicative',  fourier_order = 50)
    model.add_seasonality(name = 'daily',  period = 1, mode   = 'multiplicative',  fourier_order = 50)
    model.add_seasonality(name = 'yearly', period = 365.25, mode = 'multiplicative', fourier_order = 50)

    return model

# Function to plot performance metrics
def plot_performance_metrics(perf_metrics):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(perf_metrics['horizon'], perf_metrics['mape'], label='MAPE')
    plt.plot(perf_metrics['horizon'], perf_metrics['rmse'], label='RMSE')
    plt.xlabel('Forecast horizon')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Performance Metrics')
    st.pyplot(fig)
