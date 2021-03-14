import streamlit as st
import yfinance as yf
from datetime import date
import base64
from plotly import graph_objs as go
import seaborn as sb
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


st.write("""
         # Stock Price Predicition App
         This app predicts the **Stock Closing Price** using different Machine Learning Algorithms
         """)
st.write("""
         Author : Eric Deleon
         """)

expander_bar = st.beta_expander("About Application")
expander_bar.markdown("""
* **Python libraries:** yfinace, datetime, base64, numpy, matplotlib, seaborn, streamlit, sklearn, plotly
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

START = "2020-01-01"
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
st.subheader('Raw Data')

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

future_days = 25

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

predictions = dtr_pred
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Decision Tree Regressor')
plt.xlabel('Days')
plt.ylabel('Closing Price USD ($)')
plt.plot(df2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orginial Data', 'Validation Data', 'Predicted Value'])
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.subheader('Decision Tree Regressor: Predictive Stock Price in 25 Days')
st.text(dtr_pred[24])
st.subheader('R^2 Score')
st.text(r2_score(y_test, dtr_score))
st.subheader('RMSE')
st.text(np.sqrt(mean_squared_error(y_test,dtr_score)))

st.write('---')

predictions = rf_pred
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Random Forest Regressor')
plt.xlabel('Days')
plt.ylabel('Closing Price USD ($)')
plt.plot(df2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orginial Data', 'Validation Data', 'Predicted Value'])
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.subheader('Random Forest Regressor: Predictive Stock Price in 25 Days')
st.text(rf_pred[24])
st.subheader('R^2 Score')
st.text(r2_score(y_test, rf_score))
st.subheader('RMSE')
st.text(np.sqrt(mean_squared_error(y_test,rf_score)))

st.write('---')

predictions = lr_pred
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Linear Regression')
plt.xlabel('Days')
plt.ylabel('Closing Price USD ($)')
plt.plot(df2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orginial Data', 'Validation Data', 'Predicted Value'])
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.subheader('Linear Regression: Predictive Stock Price in 25 Days')
st.text(lr_pred[24])
st.subheader('R^2 Score')
st.text(r2_score(y_test, lr_score))
st.subheader('RMSE')
st.text(np.sqrt(mean_squared_error(y_test,lr_score)))

st.write('---')

predictions = svr_pred
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Support Vector Machines (SVM) Regressor')
plt.xlabel('Days')
plt.ylabel('Closing Price USD ($)')
plt.plot(df2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orginial Data', 'Validation Data', 'Predicted Value'])
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.subheader('SVM Regressor: Predictive Stock Price in 25 Days')
st.text(svr_pred[24])
st.subheader('R^2 Score')
st.text(r2_score(y_test, svr_score))
st.subheader('RMSE')
st.text(np.sqrt(mean_squared_error(y_test,svr_score)))










