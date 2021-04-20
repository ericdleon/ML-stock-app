#************************************************#
#              Libraries Used                    #
#                                                #
#************************************************#
# UI for application
import streamlit as st

# The Data
import yfinance as yf

# Preprocessing Data
from datetime import date
import base64
import numpy as np 
import pandas as pd

# Graphing Libraries
from plotly import graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries: Models, Splitting, Evaluating
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#************************************************#
#         Introductory to Project                #
#                                                #
#************************************************#

# "st" refers to streamlit which creates the UI
st.write("""
         # Stock Price Prediction App
         This app predicts the **Stock Closing Price** using different Machine Learning Algorithms
         """)
st.write("""
         Author : Eric Deleon
         """)

expander_bar = st.beta_expander("About the Application")
expander_bar.markdown("""
* **Python libraries:** yfinace, datetime, base64, numpy, matplotlib, seaborn, streamlit, sklearn, plotly, pandas
* **Machine Learning Techniques Used:** Decision Tree Regressor, Random Forest Regressor, Linear Regression, SVM 
* **Data source:** Yahoo! Finance
""")

expander_bar = st.beta_expander("Features in Dataset")
expander_bar.markdown("""
        * Date: The date of the trading day.
        * Open: The first trade price on Date.
        * High: The highest price at which the stock is traded on Date.
        * Low: The lowest price at which the stock is traded on Date.
        * Close: The last trade price on Date
        * Adj Close: This is defined as the closing price after all dividends are split.
        * Volume: The number of shares traded on Date. 
""")


#************************************************#
#             Fetching the Data                  #
#                                                #
#************************************************#


plt.style.use('bmh') # PLT STYLE

# Variables to Fetch Data: Between 2019 to Current Date
START = "2019-01-01"
TODAY = date.today().strftime('%Y-%m-%d')

# st.cache = cache data being looked at. This makes loading previously seen data load faster.
@st.cache
def load_data(ticker):
    # data = the stock ticker data found in Yahoo!Finace and how much of the data you want.
    data = yf.download(ticker, START, TODAY)
    # Resets the index of the data
    data.reset_index(inplace=True)
    return data # This will return the stock ticker data between 2019 and Current Date

# The sidebar where the User can interact with.
st.sidebar.subheader("""User Input Features""")
# The default stock ticker, or selected stock, that is loaded is GOOGL and the max input of a valid stock ticker is 5. Ex: VWAGY, AAPL, FB
selected_stock = st.sidebar.text_input("Enter a valid stock ticker...", "GOOG", max_chars=5)

# Load the Data
data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done")
st.subheader("""**Raw Data** for """ + selected_stock)

# Show the last items in the dataset (i.e., most recent data)
st.write(data.tail())

# This method enables the user to download the dataset of any valid stock ticker
def filedownload(df):
    # Turn df, or data, into .csv file
    csv = df.to_csv(index=False)
    # Encode in browser
    b64 = base64.b64encode(csv.encode()).decode() # strongs <-> vytes conversions
    # Create link to the .csv file
    href = f'<a href="data:file/csv;base64,{b64}" download="{selected_stock}_data.csv">Download CSV File</a>'
    return href

# Adds button that enables a user to download the data
st.markdown(filedownload(data), unsafe_allow_html=True)
st.write('---')


#************************************************#
#        Simple Moving Average Section           #
#                                                #
#************************************************#

# Copying the raw data df into another dataframe
sma = data.copy()
# The next three lines are used to get the Simple Moving Average for 5, 20, and 50 Days for the Closing Price.
sma['SMA5'] = sma.Close.rolling(5).mean()
sma['SMA20'] = sma.Close.rolling(20).mean()
sma['SMA50'] = sma.Close.rolling(50).mean()

st.subheader("""Daily **closing price** for """ + selected_stock)

# This method plots df into an interactive time series graph. 
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sma['Date'], y=sma['Close'], name='closing price'))
    fig.add_trace(go.Scatter(x=sma['Date'], y=sma['SMA5'], name='SMA5'))
    fig.add_trace(go.Scatter(x=sma['Date'], y=sma['SMA20'], name='SMA20'))
    fig.add_trace(go.Scatter(x=sma['Date'], y=sma['SMA50'], name='SMA50'))
    fig.layout.update(title_text="Time Series Data - Closing Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

st.write("""
         * SMA5 : Simple Moving Average - 5 Days
         * SMA20 : Simple Moving Avergae - 20 Days
         * SMA50 : Simple Moving Average - 50 Days
         """)

# This is used to save loading time. This is optional information that the user can load.
# If the user clicks this button, it will load the Volume Data Graph.
if st.button("""Daily Volume for """ + selected_stock):
    st.subheader("""Daily **volume** for """ + selected_stock)
    def plot_raw_volume_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name='Volume'))
        fig.layout.update(title_text="Time Series Data - Volume", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_volume_data()

st.write('---')


#************************************************#
#      Pearson Correlation & Heatmap Section     #
#                                                #
#************************************************#

if st.button("""Pearson Correlation & Heatmap for""" + selected_stock):
    st.subheader("""Pearson Correlation Coefficient for """ + selected_stock)
    corr = data.corr(method='pearson')       
    st.write(corr)
    
    st.subheader("""Correlation Heatmap for """ + selected_stock)
    st.write(sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu_r', annot=True, linewidth=0.5))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
st.write('---')    


#************************************************#
#          Machine Learning Section              #
#                                                #
#************************************************#

df = data.copy()

# If ticker data is less than 100 don't show Graphs (Ex: RBLX or COIN)
if len(df) < 100:
    st.write("Relatively new ticker, not enough data")

# Predict/forecast 31 days into the future
forecast = 31

# Keep Date and Close columns
df = df[['Date','Close']]
# Create Prediction/target column and shift 31 days up
df['Prediction'] = df[['Close']].shift(-forecast)

# Create feature dataset (X), convert to numpy array, remove the last 31 days
X = np.array(df.drop(['Prediction', 'Date'], 1))[:-forecast]
# Create target dataset (y), convert to numpy array, remove the last 31 days
y = np.array(df['Prediction'])[:-forecast]

# Get the last 31 rows of the feature dataset
x_future = df.drop(['Prediction','Date'], 1)[:-forecast]
x_future = x_future.tail(forecast)
x_future = np.array(x_future)

# Split the data to 70% Training, 30% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size=0.3, random_state=1)

# Creating the models
dtr = DecisionTreeRegressor().fit(X_train, y_train)
rf = RandomForestRegressor().fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)
svr = SVR(kernel='rbf').fit(X_train, y_train)

## Prediction Section
# Model Prediction for Decision Tree
dtr_pred = dtr.predict(x_future) # Predicting values from x_future | Visualizing
dtr_score = dtr.predict(X_test) # Evaluating the model

# Model Prediction for Random Forest
rf_pred = rf.predict(x_future) # Predicting values from x_future | Visualizing
rf_score = rf.predict(X_test) # Evaluating the model

# Model Prediction for Linear Regression
lr_pred = lr.predict(x_future) # Predicting values from x_future | Visualizing
lr_score = lr.predict(X_test) # Evaluating the model

# Model Prediction for SVM
svr_pred = svr.predict(x_future) # Predicting values from x_future | Visualizing
svr_score = svr.predict(X_test) # Evaluating the model

st.header("""**Regression Models** for """ + selected_stock)

def graph_prediction(pred, title):
    predictions = pred
    valid = df[X.shape[0]:]
    valid['Prediction'] = predictions
    plt.figure(figsize=(16,8))
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Closing Price USD ($)')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Prediction']])
    plt.legend(['Orginial Data', 'Validation Data', 'Predicted Data'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.write(valid)

def results_prediction(pred, score):
    results = [{'1-Day Forecast': pred[0],
                '1-Week Forecast': pred[6], 
                '1-Month Forecast': pred[30], 
                'R-Squared': r2_score(y_test, score),
                'MAE': mean_absolute_error(y_test, score),
                'RMSE': np.sqrt(mean_squared_error(y_test, score))
                }]
    results_df = pd.DataFrame(results)
    results_df_transposed = results_df.T
    st.write(results_df_transposed)


if st.button("""Process for Obtaining Graphs"""):
    st.write("1: Create a new df with Date and Close, Add Predictions Column and Shift 31 Days")
    st.write(df.tail(31))
    st.write("2: Breakdown df into features (X) and target (y).")
    col1, col2 = st.beta_columns(2)
    with col1:
        st.subheader("X")
        st.write(X)  
    with col2:
        st.subheader("y")
        st.write(y)
    st.write("3: Breakdown df into x_future: Get the last 31 rows of the feature dataset")
    st.write(x_future)
    st.write("4: Separate Train and Test Data")
    col3, col4 = st.beta_columns(2)
    with col3:
        st.subheader("Training Data (70%)")
        st.write("X_train")
        st.write(X_train.shape)
        st.write("y_train")
        st.write(y_train.shape)
    with col4:
        st.subheader("Testing Data (30%)")
        st.write("X_test")
        st.write(X_test.shape)
        st.write("y_test")
        st.write(y_test.shape) 
      
col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Decison Tree Regressor")
    st.write('Visual & Actual vs. Prediction Table') 
    graph_prediction(dtr_pred, 'Decison Tree Regressor')
    st.write('Day/Week/Month Forecast at-a-glance & Model Metrics') 
    results_prediction(dtr_pred, dtr_score)
    st.subheader("Linear Regression")
    st.write('Visual & Actual vs. Prediction Table') 
    graph_prediction(lr_pred, 'Linear Regression')
    st.write('Day/Week/Month Forecast at-a-glance & Model Metrics') 
    results_prediction(lr_pred, lr_score)

with col2:
    st.subheader("Random Forest Regressor")
    st.write('Visual & Actual vs. Prediction Table') 
    graph_prediction(rf_pred, 'Random Forest Regressor')
    st.write('Day/Week/Month Forecast at-a-glance & Model Metrics') 
    results_prediction(rf_pred, rf_score)
    st.subheader("SVM Regressor")
    st.write('Visual & Actual vs. Prediction Table') 
    graph_prediction(svr_pred, 'Support vector Machines (SVM) Regressor')
    st.write('Day/Week/Month Forecast at-a-glance & Model Metrics') 
    results_prediction(svr_pred, svr_score)

if st.button('Large Graph Option - For PC Screens'):
    graph_prediction(dtr_pred, 'Decison Tree Regressor')
    results_prediction(dtr_pred, dtr_score)
    st.write('---')
    graph_prediction(rf_pred, 'Random Forest Regressor')
    results_prediction(rf_pred, rf_score)
    st.write('---')
    graph_prediction(lr_pred, 'Linear Regression')
    results_prediction(lr_pred, lr_score)
    st.write('---')
    graph_prediction(svr_pred, 'Support Vector Machines (SVM) Regressor')
    results_prediction(svr_pred, svr_score)