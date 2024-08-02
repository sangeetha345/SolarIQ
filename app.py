from pyspark.sql import SparkSession
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing 

# Create a SparkSession
spark = SparkSession.builder \
    .appName("EnergyDataAnalysis") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Path to the CSV file
file_path = "/workspaces/Solar-Power-Estimator/96-Site_DKA-MasterMeter1.csv"

# Load the CSV file into a DataFrame
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Show the first few rows of the DataFrame
df.show()

# Display schema
df.printSchema()

# Perform any additional processing here
# For example, you can perform some basic analysis
df.describe().show()

# Get the number of rows
num_rows = df.count()

# Get the number of columns
num_columns = len(df.columns)

# Print the shape of the DataFrame
print(f"DataFrame shape: ({num_rows}, {num_columns})")

# Convert to Pandas DataFrame for easier manipulation
pdf = df.toPandas()

# Streamlit Sidebar
st.sidebar.title("Energy Data Analysis and Forecasting")
page = st.sidebar.selectbox("Choose a page", ["EDA", "Forecasting"])

if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Summary statistics
    st.header("Summary Statistics")
    st.write(pdf.describe())

    # Plot distributions
    st.header("Distributions")
    columns = pdf.columns
    for col in columns[1:]:
        fig, ax = plt.subplots()
        sns.histplot(pdf[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Correlation matrix
    st.header("Correlation Matrix")
    corr = pdf.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Time series plot
    st.header("Time Series Plot")
    time_series_cols = ["Active_Energy_Delivered_Received", "Active_Power"]
    fig = px.line(pdf, x="timestamp", y=time_series_cols)
    st.plotly_chart(fig)

elif page == "Forecasting":
    st.title("Time Series Forecasting")

    # Select a column for forecasting
    target_col = st.selectbox("Select a column to forecast", pdf.columns[1:])

    # Prepare data for forecasting
    data = pdf[["timestamp", target_col]].dropna()
    data.set_index("timestamp", inplace=True)
    data.index = pd.to_datetime(data.index)

    # Split data into training and test sets
    train_data = data[:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]

    # Train the model
    model = ExponentialSmoothing(train_data[target_col], trend="add", seasonal="add", seasonal_periods=12).fit()

    # Forecast
    forecast = model.forecast(steps=len(test_data))

    # Plot the results
    fig, ax = plt.subplots()
    train_data[target_col].plot(ax=ax, label="Train")
    test_data[target_col].plot(ax=ax, label="Test")
    forecast.plot(ax=ax, label="Forecast", linestyle="--")
    ax.legend()
    st.pyplot(fig)

# Stop the SparkSession
spark.stop()
