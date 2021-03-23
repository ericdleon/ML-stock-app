import streamlit as st
import yfinance as yf
from datetime import date
import base64
from plotly import graph_objs as go
import seaborn as sb
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler


st.write("""
         # Stock Price Predicition App
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

plt.style.use('bmh')

START = "2019-01-01"
TODAY = date.today().strftime('%Y-%m-%d')

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.sidebar.subheader("""User Input Features""")
selected_stock = st.sidebar.text_input("Enter a valid stock ticker...", "GOOG", max_chars=5)

data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done")
st.subheader("""**Raw Data** for """ + selected_stock)

st.write(data.tail())

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strongs <-> vytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{selected_stock}_data.csv">Download CSV File</a>'
    return href
st.markdown(filedownload(data), unsafe_allow_html=True)
st.write('---')

df = data.copy()
df['SMA5'] = df.Close.rolling(5).mean()
df['SMA20'] = df.Close.rolling(20).mean()
df['SMA50'] = df.Close.rolling(50).mean()

st.subheader("""Daily **closing price** for """ + selected_stock)
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='closing price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA5'], name='SMA5'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], name='SMA20'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], name='SMA50'))
    fig.layout.update(title_text="Time Series Data - Closing Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

st.write("""
         * SMA5 : Simple Moving Average - 5 Days
         * SMA20 : Simple Moving Avergae - 20 Days
         * SMA50 : Simple Moving Average - 50 Days
         """)

if st.button("""Daily Volume for """ + selected_stock):
    st.subheader("""Daily **volume** for """ + selected_stock)
    def plot_raw_volume_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name='Volume'))
        fig.layout.update(title_text="Time Series Data - Volume", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_volume_data()

st.write('---')

st.subheader("""Pearson Correlation Coefficient for """ + selected_stock)
corr = data.corr(method='pearson')       
st.write(corr)

st.subheader("""Confusion Matrix for """ + selected_stock)
st.write(sb.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu_r', annot=True, linewidth=0.5))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.write('---')

df2 = data.copy()

df2 = df2[['Close']]
future_days = 31
df2['Prediction'] = df2[['Close']].shift(-future_days)

X = np.array(df2.drop(['Prediction'], 1))[:-future_days]
y = np.array(df2['Prediction'])[:-future_days]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size=0.3, random_state=0)

dtr = DecisionTreeRegressor().fit(X_train, y_train)
rf = RandomForestRegressor().fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)
svr = SVR(kernel='rbf').fit(X_train, y_train)

x_future = df2.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

dtr_pred = dtr.predict(x_future)
dtr_score = dtr.predict(X_test)
rf_pred = rf.predict(x_future)
rf_score = rf.predict(X_test)
lr_pred = lr.predict(x_future)
lr_score = lr.predict(X_test)
svr_pred = svr.predict(x_future)
svr_score = svr.predict(X_test)

st.header("""**Regression Models** for """ + selected_stock)

def graph_prediction(pred, title):
    predictions = pred
    valid = df2[X.shape[0]:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16,8))
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Closing Price USD ($)')
    plt.plot(df2['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Orginial Data', 'Validation Data', 'Predicted Data'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def results_prediction(pred, score):
    results = [{'1-Day Forecast': pred[0],'1-Week Forecast': pred[6], '1-Month Forecast': pred[30], 'R-Squared Score': r2_score(y_test, score), 'RMSE': np.sqrt(mean_squared_error(y_test, score))}]
    results_df = pd.DataFrame(results)
    results_df_transposed = results_df.T
    st.write(results_df_transposed)

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Decison Tree Regressor")
    graph_prediction(dtr_pred, 'Decison Tree Regressor')
    results_prediction(dtr_pred, dtr_score)
    st.subheader("Linear Regression")
    graph_prediction(lr_pred, 'Linear Regression')
    results_prediction(lr_pred, lr_score)

with col2:
    st.subheader("Random Forest Regressor")
    graph_prediction(rf_pred, 'Random Forest Regressor')
    results_prediction(rf_pred, rf_score)
    st.subheader("SVM Regressor")
    graph_prediction(svr_pred, 'Support vector Machines (SVM) Regressor')
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