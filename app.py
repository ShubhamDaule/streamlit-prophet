import streamlit as st
import plotting as plotter
import pandas as pd
from prophet import Prophet
import combined_bayes as pm
import io
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import datetime
import math
import pickle
import plotly.graph_objects as go
import plotly.graph_objs as go
import numpy as np
from utils.load import load_image,load_toy_dataset
from utils.models import prophet_modeling,plot_performance_metrics, load_data#, risk_assessment
st.title('DataGush')
st.sidebar.title('Settings')

################################################################################
#                       DATA
################################################################################
st.sidebar.title("1. Data")
df = None
# Sidebar for dataset selection
with st.sidebar.expander("Dataset", expanded=True):
    option = st.selectbox("Choose an option", ("Use preloaded data", "Upload your own"))
    if option == 'Use preloaded data':
        choices = ["3DS 2.0", "CGK1"]
        AppName = st.sidebar.selectbox("Select an option", choices, index = 0)
        default_sheet_name = str(AppName) + " Hourly Data"
        default_file_path = "/var/MLP/Dev/ApplicationHourlyVolume.xlsx"
        df = load_data(default_file_path, default_sheet_name)
        df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.rename(columns={'Date': "ds", 'Volume': "y"})
        holidays_data_path = "/var/MLP/Dev/holidays.xlsx"
        holidays_df1 = pd.read_excel(holidays_data_path)


        # Get excel data and format
        inputData = pd.ExcelFile("/var/MLP/Dev/CapacityInput.xlsx")
        try:
            serviceCapacityData = pd.read_excel(inputData, AppName + " Services")
        except:
            #if capacity data not available set to 0 so forecasting can continue
            serviceCapacityData = {'Services': [AppName], 'No. of Nodes': [0], 'Capacity/Node': [0], 'Current Capacity TPS': [0], 'Percentage': [1]}
        applicationData = pd.read_excel(inputData, "Applications")
       
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx", "xls"], help="Upload your Excel dataset", accept_multiple_files=False)
        if uploaded_file is not None:
            sheet_name = st.selectbox("Select a sheet", pd.ExcelFile(uploaded_file).sheet_names, help="Select a sheet from the uploaded Excel file")
            df = load_data(uploaded_file, sheet_name)
            # Column names
            if df is not None:
                with st.sidebar.expander("Columns", expanded=True):
                    date_col = st.selectbox("Date column",sorted(df.columns))
                    target_col = st.selectbox( "Target column", sorted(set(df.columns) - {date_col}) )
                    df = df.rename(columns={date_col: "ds", target_col: "y"})
            else: 
                st.write("Please upload an excel file")
                st.stop()

################################################################################
#                       HOLIDAYS
################################################################################
st.sidebar.title("2. Holidays")
# Custom Holidays data
with st.sidebar.expander("Custom Holidays"):
    holidays_data = st.file_uploader(label="Upload a excel file with holiday and dates", type=["xlsx", "xls"], help="holiday_data_upload", accept_multiple_files=False)
    if holidays_data is not None:
        holidays_df = pd.read_excel(holidays_data)
    else:
        holidays_df = None

################################################################################
#                       FORECASTING
################################################################################
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name ='Sheet1')
    processed_data = output.getvalue() 
    return processed_data
st.sidebar.title("3. Forecast")
numMonths = st.sidebar.number_input("Number of Periods(Months)", min_value=1, value=1)
view_error = st.sidebar.checkbox("Show error metrics", value = False)
show_plots = st.sidebar.checkbox("Show plots", value=True)
#download = st.sidebar.checkbox("Download forecast when complete", value = False)
generate_button = st.sidebar.button("Generate Forecast")
#forecast_data = st.sidebar.checkbox("Launch Forecast", value=True)
if generate_button:

    if option == 'Use preloaded data':
        holidays_df = holidays_df1
        if holidays_data is not None:
            holidays_df = holidays_df
        #st.write('Run predefined model')
        if AppName == '3DS 2.0':
            filename_3ds = "/var/MLP/Dev/pickle_3ds.sav"
            loaded_model = pickle.load(open(filename_3ds, 'rb')) 
            model = loaded_model
        else:
            # Perform modeling
            model = prophet_modeling(holidays_df)
            model.fit(df)
    else:
        
        model = pm.predict(df, holidays_df)

    
    firsttwo = df['ds'].iloc[:2]
    timedif = (firsttwo[1] - firsttwo[0]).total_seconds()
    
    if timedif <= 3600:
        #st.write('hourly data')
        future= model.make_future_dataframe(periods=730*numMonths,freq='h')
    elif timedif <= 86400:
        future= model.make_future_dataframe(periods=30*numMonths,freq='D')
    elif timedif >=2419200:
        future= model.make_future_dataframe(periods=numMonths,freq='M')
    else:
        future= model.make_future_dataframe(periods=730*numMonths,freq='h')
    #future = model.make_future_dataframe(periods=730*2,freq='H')
    forecast = model.predict(future)
    
    st.subheader('Forecast Results')  
    
    st.write(forecast)
    
    ypred = forecast[:df.shape[0]]

    
    if show_plots:
        # Plot forecast
        st.subheader('Forecast')
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        fig_AN = plotter.plot_AN(model, forecast)
        st.plotly_chart(fig_AN)    
        # Plot components
        st.subheader('Components')
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
    
    
        # import streamlit as st
        # from datetime import datetime, timedelta
    
        fig=go.Figure()
        
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast['yhat_upper'], name="Forecast_Upper", mode="lines", line_color="lightblue"))
        # fig.add_trace(go.Scatter(x=ypred["ds"], y=ypred['yhat'], name="Predicted", mode="lines", line_color="blue"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast['yhat'], name="Forecast", mode="lines", line_color="blue"))
        fig.add_trace(go.Scatter(x=df["ds"], y=df['y'], name="Actual", mode="lines", line_color="green"))
    
        fig.update_layout(
            title= "Hourly Volume Forecast", xaxis_title="Date", yaxis_title="Volume"
        )
    
        st.plotly_chart(fig)
        
#################################################################################
##                       ERROR
#################################################################################
 #   if download:
 #       excel_file = to_excel(forecast)
 #       st.download_button(label="Download forecast data as Excel",
 #                               data=excel_file,
 #                               file_name = 'forecast_data.xlsx',
 #                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
 
    if view_error and generate_button:
        error = pm.evaluate_model(model)
        st.write("MAPE: ", error)


#################################################################################
#                            Service Risk Assesment  
#################################################################################

    st.sidebar.title("4. Service Risk Assesment")
    
    responseTime = st.sidebar.number_input("Expected Response Time", min_value=0.0, max_value= 2.0, value=0.0, step=0.1)
    expectedTPS = st.sidebar.number_input("Expected TPS", min_value=0, value=0)
    #st.write("Response Time: ", responseTime)
    
    assesment = st.sidebar.checkbox("Peak Hour Forecast Calculation", value=False)
    
    if assesment:
        #st.subheader('Current capacity of Application specified in excel')
        
        #Volume Forecast From Prophet Script
        # ForecastData = pd.read_excel(r'ForecastOutput.xlsx')
        ForecastData = forecast
        #st.write(AppName)
        #st.write(applicationData.head())
    
        totalCapacity = applicationData.loc[applicationData['Application'] == AppName]['Current Capacity - TPS'].item()
        # totalCapacity = int(re.sub("[^0-9]", "", totalCapacity)) #remove spaces and TPS
    
        # Pull forcasted data
        # Forecasted data generated from running forecasting script first
        data2 = ForecastData[['ds', 'yhat_lower','yhat_upper','yhat']]
    
        #determining average peak hour TPS
        PeakHourAverage = ForecastData.loc[:, 'yhat']
    
        data2.loc[:,'AveragePeakHour'] = PeakHourAverage/3600
    
        PeakHour_Lower = ForecastData.loc[:, 'yhat_lower']
        data2.loc[:, 'AveragePeakHour_Lower'] = PeakHour_Lower/3600
    
        PeakHour_Upper = ForecastData.loc[:, 'yhat_upper']
        data2.loc[:, 'AveragePeakHour_Upper'] = PeakHour_Upper/3600
    
        history_PeakHour = train.loc[:, 'y']
        train.loc[:, 'AveragePeakHour'] = history_PeakHour/3600
    
        st.subheader('Current Capacity of Application For Peak Hour')
        st.write("Total Current Capacity of %s: %d" % ( AppName, totalCapacity))
    
        multiplier = 2
        data2.loc[:, 'PeakHourForecast'] = data2.loc[:, 'AveragePeakHour'] * multiplier
        data2.loc[:, 'PeakHourForecast_Upper'] = data2.loc[:, 'AveragePeakHour_Upper'] * multiplier
        data2.loc[:, 'PeakHourForecast_Lower'] = data2.loc[:, 'AveragePeakHour_Lower'] * multiplier
        train.loc[:, 'PeakHourForecast'] = train.loc[:, 'AveragePeakHour'] * multiplier
    
        certResponseTime = applicationData.loc[applicationData['Application'] == AppName]['Certified Response Time (s)'].item()
    
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data2["ds"], y=data2.loc[:,'PeakHourForecast_Lower'], name="PeakHourForecast_Lower", mode="lines", line_color="lightblue"))
        fig.add_trace(go.Scatter(x=data2["ds"], y=data2.loc[:,'PeakHourForecast_Upper'], name="PeakHourForecast_Upper", mode="lines", line_color="lightblue"))
        fig.add_trace(go.Scatter(x=data2["ds"], y=data2.loc[:,'PeakHourForecast'], name="PeakHourForecast", mode="lines", line_color='blue'))
        fig.add_trace(go.Scatter(x=train["ds"], y=train.loc[:,'PeakHourForecast'], name="PeakHourForecast_History", mode="lines", line_color='lightgreen'))
        # fig.add_trace(go.Scatter(x=data2["ds"], y=data2.loc[:,'PeakHourForecast_Upper'], name="PeakHourForecast_upper", mode="lines", line_color="lightblue"))
        fig.add_hline(y=totalCapacity, line_color="red", annotation_text="Current Capacity", annotation_position="top left")
        # fig.add_hline(y=totalCapacity/responseTimeIncrease, line_color="yellow", annotation_text="Capacity with " + str(responseTimeIncrease) + "x response times", annotation_position="top left")
        if responseTime > certResponseTime:
            fig.add_hline(y=totalCapacity/(responseTime/certResponseTime), line_color="yellow", annotation_text="Capacity with " + str(responseTime) + "s response times", annotation_position="top left")

        # fig.add_hline(y=totalCapacity/2, line_dash="dash", line_color="red", annotation_text="50% Capacity", annotation_position="top left")
        fig.update_layout(
            title="Peak " + AppName + " TPS", xaxis_title="Date", yaxis_title="TPS"
        )
        # fig.update_xaxes(range=[data['ds'].max()-timedelta(days=30), pred_CI95['ds'].max()])
        st.plotly_chart(fig)
    
    
        st.subheader('Forecast Over Capacity Limit For Application')
        column_name = 'PeakHourForecast_Upper'
        column = data2.loc[:, column_name]
        count = column[column > totalCapacity].count()
        st.write("Number of forecasted breaches of current capacity: %d" % count)
    
    
        maxTPS = data2[column_name].max()
    
        totalCapPer = applicationData.loc[applicationData['Application'] == AppName]['Capacity Per Server'].item()
        totalServers = applicationData.loc[applicationData['Application'] == AppName]['Current No. of servers'].item()
    
        st.markdown(f"## <span style='color:blue'>{AppName}</span>", unsafe_allow_html=True)
        st.write(f"Current capacity: {totalCapacity} TPS")
        st.markdown(f"Max TPS: <span style='color:red'>{round(maxTPS, 2)}</span>", unsafe_allow_html=True)
    
        if totalCapPer != 0:
            newNodes = math.ceil(maxTPS / totalCapPer) - totalServers
        else:
            st.error("Error: The 'totalCapPer' variable is zero, cannot calculate 'newNodes'.")
    
        tpsIncrease = maxTPS - totalCapacity
    
        #new_capacity = st.text_input("Enter the updated capacity after adding new hardware: ")
        new_capacity = st.sidebar.number_input("Updated Capacity With New Hardware", min_value=0, value=0)
        
        if new_capacity:
            new_capacity = int(new_capacity)
            new_breach_count = column[column > new_capacity].count()
            st.write(f"Number of forecasted breaches of new capacity: {new_breach_count}")
    
            new_max_TPS = data2[column_name].max()
            new_nodes = math.ceil(new_max_TPS / totalCapPer) - totalServers
    
            break_date = "The new capacity is not expected to be exceeded within the forecast period."
            for index, row in data2.iterrows():
                if row[column_name] > new_capacity:
                    break_date = row['ds']
                    break

    
            st.write(f"The new capacity can sustain the growth until: {break_date}")
    
            try:
                nodeCPU = applicationData.loc[applicationData['Application'] == AppName]['CPU'].item()
            except KeyError:
                nodeCPU = 0
    
            try:
                nodeMem = applicationData.loc[applicationData['Application'] == AppName]['Memory'].item()
            except KeyError:
                nodeMem = 0

    
            if tpsIncrease >= 0:
                st.markdown(f"Max TPS is <span style='color:red'>{round(tpsIncrease)} TPS</span> above current capacity", unsafe_allow_html=True)
                st.markdown(f"Number of additional instances needed: <span style='color:red'>{int(newNodes)}</span>", unsafe_allow_html=True)
                if nodeCPU > 0:
                    st.markdown(f"Total additional CPUs required: <span style='color:red'>{math.ceil(newNodes*nodeCPU)}</span>", unsafe_allow_html=True)
                if nodeMem > 0:
                    st.markdown(f"Total additional memory required: <span style='color:red'>{math.ceil(newNodes*nodeMem)}GB</span>", unsafe_allow_html=True)
            elif tpsIncrease < 0:
                st.markdown(f"Max TPS is <span style='color:red'>{round(tpsIncrease)} TPS</span> below current capacity", unsafe_allow_html=True)
                st.write("No additional instances needed")

    
    
        # set risk to true for any date/time where peak forecast is over current capacity   
        data2.loc[:, 'RiskPresent'] = np.where(data2.loc[:, column_name] > totalCapacity, True, False)
    
    
    
    
        analysis = st.sidebar.checkbox("Risk Analysis for Services", value=False)
    
        if analysis:
            st.subheader('Percentage of Overall Volume And Current Capacity For Each Service')

            st.dataframe(serviceCapacityData.head(100))
        
            # Set tps capacity per component based on percentage of total load.
            # If no percentage is available or if less than 10% set forecast to 10% of total load
        
            for index, row in serviceCapacityData.iterrows():
                name = row['Services']
                percentage = row.loc['Percentage']
                if math.isnan(percentage): 
                    st.write(f"{name} does not have breakdown percentage, setting to 10%")
                    percentage = 0.1
        
                data2[f"{name}_Forecast"] = data2.loc[:, column_name] * percentage

    
    
            for index, row in serviceCapacityData.iterrows():
                name = row['Services']
                riskName = name + "_RiskPresent"
                tps = row['Current Capacity TPS']
                if math.isnan(tps):
                    st.write(name, "has no TPS capacity info")
                    tps = 0
                dataRisk = np.where(data2[name+"_Forecast"] > tps, True, False)
                data2[riskName] = dataRisk
    
            risks = data2.filter(like='_RiskPresent').apply(lambda row: row[row==True], axis=1)

    
            #if risks.empty:
            #    st.markdown("<span style='color:red'>No at risk services</span>", unsafe_allow_html=True)
            #for i in risks:
            #    service = i.replace("_RiskPresent", "")
            #    riskF = i.replace("_RiskPresent", "_Forecast")
            #    for index, row in serviceCapacityData.iterrows():
            #        name = row['Services']
            #        if(name==service):
            #            capPer = row['Capacity/Node']
            #            currNodes = row['No. of Nodes']
            #            currCap = row['Current Capacity TPS']
            #            try:
            #                nodeCPU = row['CPU']
            #            except KeyError:
            #                nodeCPU = 0
            #            try:
            #                nodeMem = row['Memory']
            #            except KeyError:
            #                nodeMem = 0

            #    for row, ind in data2.iterrows():
            #        if ind[riskF] > currCap:
            #            breakTPS = ind[riskF]
            #            breakDate = ind["ds"]
            #            if breakDate > datetime.now():
            #                st.markdown(f"<span style='color:blue'><b>{service}:</b></span>", unsafe_allow_html=True)
            #                st.write(f"Current capacity {currCap} TPS")
            #                st.markdown("Service will hit <span style='color:red'>{round(breakTPS, 2)} TPS</span> at <span style='color:red'>{breakDate}</span>", unsafe_allow_html=True)
            #                newNodes = math.ceil(breakTPS / capPer) - currNodes
            #                st.write(f"Additional instances needed: {newNodes}")

            #                if nodeCPU > 0:
            #                    st.markdown(f"Total additional CPUs required: <span style='color:red'>{math.ceil(newNodes*nodeCPU)}</span>", unsafe_allow_html=True)
            #                if nodeMem > 0:
            #                    st.markdown(f"Total additional memory required: <span style='color:red'>{math.ceil(newNodes*nodeMem)}GB</span>", unsafe_allow_html=True)
            #                st.write("")
            #                break

    
            st.subheader('Max TPS for All Services')
    
            for i in serviceCapacityData['Services']:
                service = i
                riskF = i + "_Forecast"
                serviceMax = data2[riskF].max()
                for index, row in serviceCapacityData.iterrows():
                    name = row['Services']
                    if(name==service):
                        capPer = row['Capacity/Node']
                        currNodes = row['No. of Nodes']
                        currCap = row['Current Capacity TPS']
                        try:
                            nodeCPU = row['CPU']
                        except KeyError:
                            nodeCPU = 0
                        try:
                            nodeMem = row['Memory']
                        except KeyError:
                            nodeMem = 0

    
                        try:
                            k8 = row['K8']
                        except KeyError:
                            k8 = 'n'
                        if k8 == 'y':
                            podMem = row['pod mem limit']
                            podCpu = row['pod cpu limit']
                            nodeMem = row['node mem']
                            nodeCpu = row['node cpu']
    
                st.markdown(f"<span style='color:blue'><b>{service}:</b></span>", unsafe_allow_html=True)
                st.write(f"Current capacity: {currCap} TPS")
                st.markdown(f"Max forecasted TPS: <span style='color:red'>{round(serviceMax, 2)}</span>", unsafe_allow_html=True)

    
                newNodes = math.ceil(serviceMax / capPer) - currNodes
                tpsIncrease = serviceMax - currCap
                if tpsIncrease >= 0:
                    if k8 == 'y':
                        totalMem = newNodes * podMem
                        totalCpu = newNodes * podCpu
                        numNeed = max(totalMem / nodeMem, totalCpu / nodeCpu)      
                        st.markdown(f"Number of additional pods needed: <span style='color:red'>{int(newNodes)}</span>", unsafe_allow_html=True)
                        st.markdown(f"Number of additional servers needed to support new pods: <span style='color:red'>{math.ceil(numNeed)}</span>", unsafe_allow_html=True)
                    else:
                        st.write(f"Max forecasted TPS is <span style='color:red'>{round(tpsIncrease)} TPS</span> above current capacity", unsafe_allow_html=True)
                        st.markdown(f"Number of additional nodes needed: <span style='color:red'>{newNodes}</span>", unsafe_allow_html=True)
                        if nodeCPU > 0:
                            st.markdown(f"Total additional CPUs required: <span style='color:red'>{math.ceil(newNodes*nodeCPU)}</span>", unsafe_allow_html=True)
                        if nodeMem > 0:
                            st.markdown(f"Total additional memory required: <span style='color:red'>{math.ceil(newNodes*nodeMem)}GB</span>", unsafe_allow_html=True)
                elif tpsIncrease < 0:
                    st.write(f"Max forecasted TPS is <span style='color:red'>{round(tpsIncrease)} TPS</span> below current capacity", unsafe_allow_html=True)
                    st.write("No additional nodes needed")
                st.write("")

    
    
        withRisk = st.sidebar.checkbox("Forecasted TPS Graphs for at Risk Services", value=False)
        if withRisk:
            st.subheader("Forecasted TPS Graphs for at Risk Services")
            for i in risks:
                service = i.replace("_RiskPresent", "")
                riskF = i.replace("_RiskPresent", "_Forecast")    
                for index, row in serviceCapacityData.iterrows():
                    name = row['Services']
                    if(name==service):
                        capPer = row['Capacity/Node']
                        currNodes = row['No. of Nodes']
                        currCap = row['Current Capacity TPS']
                        try:
                            certResponseTime = row['Certified Response Time (s)']
                        except KeyError:
                            certResponseTime = 0

                fig=go.Figure()
                fig.add_trace(go.Scatter(x=data2["ds"], y=data2[riskF], name=riskF, mode="lines", line_color='blue'))
                fig.add_hline(y=currCap, line_color="red", annotation_text="Current Capacity", annotation_position="top left")
                if responseTime > certResponseTime:
                    fig.add_hline(y=currCap/(responseTime/certResponseTime), line_color="yellow", annotation_text="Capacity with " + str(responseTime) + "s response times", annotation_position="top left")
    
                fig.update_layout(
                    title=service.capitalize(), xaxis_title="Date", yaxis_title="TPS"
                )
                fig.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig)

        withoutRisk = st.sidebar.checkbox("Forecasted TPS Graphs For Services Without Risk", value=False)        
        if withoutRisk:
            st.subheader("Forecasted TPS Graphs for at Risk Services")
            services = data2.filter(like='_Forecast')
            haveRisk = []
    
            for i in risks:
                service = i.replace("_RiskPresent", "")
                haveRisk.append(service)
    
            # Check if haveRisk is empty
            if not haveRisk:
                st.write("No services with risk")
    
            for i in services:
                service = i.replace("_Forecast", "")
                if service not in haveRisk:
                
                    for index, row in serviceCapacityData.iterrows():
                        name = row['Services']
                        if(name==service):
                            capPer = row['Capacity/Node']
                            currNodes = row['No. of Nodes']
                            currCap = row['Current Capacity TPS']
                            try:
                                certResponseTime = row['Certified Response Time (s)']
                            except KeyError:
                                certResponseTime = 0

                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=data2["ds"], y=data2[i], name=service, mode="lines"))
                    fig.add_hline(y=currCap, line_color="red", annotation_text="Current Capacity", annotation_position="top left")
                    if responseTime > certResponseTime:
                        fig.add_hline(y=currCap/(responseTime/certResponseTime), line_color="yellow", annotation_text="Capacity with " + str(responseTime) + "s response times", annotation_position="top left")
                    fig.update_layout(
                        title=service.capitalize(), xaxis_title="Date", yaxis_title="TPS"
                    )
                    st.plotly_chart(fig)

    
        addInfra = st.sidebar.checkbox("Additional Infrastructure to Meet Expected TPS", value=False)
        #withRisk = st.subheader("Forecasted TPS Graphs for at Risk Services", value=False)
        if addInfra:
            if 'expectedTPS' in locals():
                if expectedTPS > 0:
                    SHOW = True
                else:
                    SHOW = False
            else:
                SHOW = False
    
            additional_infrastructure_needed = False
    
            if SHOW:
                st.markdown("""### Additional Infrastructure to Meet Expected TPS""")

    
                #...
    
                st.write(f'**{AppName}:**')
                st.write(f"Current capacity: {totalCapacity} TPS")
                st.write(f"Expected TPS: {round(expectedTPS, 2)}")
    
                #...
    
                for index, row in serviceCapacityData.iterrows():
                    service = row['Services']
                    capPer = row['Capacity/Node']
                    currNodes = row['No. of Nodes']
                    currCap = row['Current Capacity TPS']
                    percentage = row['Percentage']
                    tpsExpected = percentage * expectedTPS
    
                    st.write(f'**{service}:**')
                    st.write(f"Current capacity: {currCap} TPS")
                    st.write(f"Expected TPS: {round(tpsExpected, 2)}")

    
                    newNodes = math.ceil(tpsExpected / capPer) - currNodes
                    if newNodes > 0:
                        additional_infrastructure_needed = True
                    #...
    
            if not additional_infrastructure_needed:
                st.write("No additional infrastructure is needed.")
