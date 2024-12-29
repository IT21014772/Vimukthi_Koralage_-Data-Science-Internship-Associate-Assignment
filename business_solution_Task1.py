import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime

# Function to compute RFM metrics
def compute_rfm(data):
    rfm = data.groupby('customerId').agg({
        'recency_days': 'min',
        'num_transactions': 'sum',
        'total_amount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

# RFM Segmentation
def rfm_segmentation(rfm):
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm

# Streamlit app
def main():
    st.title("Supermarket Customer Segmentation App")

    # File upload section
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Preprocess date columns
        data['first_purchase_date'] = pd.to_datetime(data['first_purchase_date'], errors='coerce')
        data['last_purchase_date'] = pd.to_datetime(data['last_purchase_date'], errors='coerce')

        # Sidebar Navigation
        st.sidebar.header("Navigation")
        options = ["Overview", "Customer Insights", "RFM Segmentation"]
        choice = st.sidebar.selectbox("Choose a section", options)

        if choice == "Overview":
            st.header("Dataset Overview")
            st.write("Here is a preview of the dataset:")
            st.dataframe(data.head())

            st.write("Summary Statistics (Excluding Customer ID):")
            numeric_data = data.select_dtypes(include=['number']).drop(columns=['customerId'], errors='ignore')
            st.write(numeric_data.describe())

        elif choice == "Customer Insights":
            st.header("Customer Insights")

            # Total sales and returns
            total_sales = data[data['transaction_type'] == 'sale']['total_amount'].sum()
            total_returns = data[data['transaction_type'] == 'return']['total_amount'].sum()
            st.metric("Total Sales", f"${total_sales:,.2f}")
            st.metric("Total Returns", f"${total_returns:,.2f}")

            # Transactions by day of the week
            st.subheader("Transactions by Day of the Week")
            fig, ax = plt.subplots()
            sns.countplot(x='day_of_week', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax)
            st.pyplot(fig)

            # Hourly transaction trends
            st.subheader("Hourly Transaction Trends")
            fig, ax = plt.subplots()
            sns.histplot(data['hour'], bins=24, kde=False, ax=ax)
            ax.set_xlabel("Hour of the Day")
            ax.set_ylabel("Number of Transactions")
            st.pyplot(fig)

        elif choice == "RFM Segmentation":
            st.header("RFM Segmentation")

            # Compute RFM metrics
            rfm = compute_rfm(data)
            rfm = rfm_segmentation(rfm)

            st.write("Customer Segmentation based on RFM analysis:")
            st.dataframe(rfm)

            st.subheader("RFM Distribution")
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            sns.histplot(rfm['Recency'], kde=True, ax=ax[0])
            ax[0].set_title("Recency Distribution")

            sns.histplot(rfm['Frequency'], kde=True, ax=ax[1])
            ax[1].set_title("Frequency Distribution")

            sns.histplot(rfm['Monetary'], kde=True, ax=ax[2])
            ax[2].set_title("Monetary Distribution")

            st.pyplot(fig)

            # Clusters Visualization
            st.subheader("Customer Segments")
            fig, ax = plt.subplots()
            sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Segment', palette='viridis', ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
