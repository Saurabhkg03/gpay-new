import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup

def extract_receiver(content):
    receiver_match = re.search(r'to\s+(\S[\w\s]+)\s+using', content)
    if receiver_match:
        return receiver_match.group(1)
    else:
        return None

def classify_with_keywords(receiver_name, predicted_label):
    # Define keyword lists for different transaction types
    food_keywords = ['zomato', 'swiggy', 'ubereats', 'foodpanda']
    clothing_keywords = ['nike', 'adidas', 'zara', 'h&m']
    groceries_keywords = ['walmart', 'tesco', 'bigbasket', 'grofers']

    # Check if the receiver's name contains any keywords
    if any(keyword in receiver_name.lower() for keyword in food_keywords):
        return 'Food'
    elif any(keyword in receiver_name.lower() for keyword in clothing_keywords):
        return 'Clothing'
    elif any(keyword in receiver_name.lower() for keyword in groceries_keywords):
        return 'Groceries'
    else:
        return predicted_label

def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    divs = soup.find_all('div', class_='outer-cell')
    data = []
    for div in divs:
        description = div.find('p', class_='mdl-typography--title').text.strip()
        content = div.find('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1').text.strip()
        receiver = extract_receiver(content)
        if receiver is None:
            continue  # Skip divs without receiver information
        amount = content.split()[1]
        date_match = re.search(r'(\w{3} \d{1,2}, \d{4}, \d{1,2}:\d{1,2}:\d{1,2})', content)
        if date_match:
            date = date_match.group(1)
        else:
            date = ''
        data.append({'Description': description, 'Amount': amount, 'Receiver': receiver, 'Date': date})
    return pd.DataFrame(data)

st.title("Transaction Data Processor")

uploaded_file = st.file_uploader("Upload HTML file", type=["html"])

if uploaded_file is not None:
    html_content = uploaded_file.read()
    df = extract_data(html_content)

    # Load the Hugging Face model for transaction type classification
    tokenizer = AutoTokenizer.from_pretrained("mgrella/autonlp-bank-transaction-classification-5521155")
    model = AutoModelForSequenceClassification.from_pretrained("mgrella/autonlp-bank-transaction-classification-5521155")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Predict transaction types
    receiver_names = df['Receiver'].tolist()
    predictions = pipe(receiver_names)

    # Get the predicted label for each transaction and remove "Category." part
    predicted_labels = [prediction['label'].replace("Category.", "") for prediction in predictions]

    # Apply keyword-based classification
    df['Transaction Type'] = [classify_with_keywords(receiver_names[i], predicted_labels[i]) for i in range(len(df))]

    # Export data to Excel and remove "Category." part from Transaction Type column
    df['Transaction Type'] = df['Transaction Type'].str.replace("Category.", "")

    # Save DataFrame to Excel without wrapping text
    excel_file_path = 'extracted_data_with_transaction_type_and_keywords.xlsx'
    df.to_excel(excel_file_path, index=False)
    
    # Provide download link to the processed data in Excel format
    st.download_button(
        label="Download processed data",
        data=open(excel_file_path, 'rb'),
        file_name=excel_file_path,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.success('Data extracted and download button added')

# Convert 'Amount' column to float
df['Amount'] = df['Amount'].str.replace('₹', '').str.replace(',', '').astype(float)

# Clean and convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y, %H:%M:%S')

# Define functions for analysis

def overview_section():
    total_transactions = len(df)
    total_amount_spent = df['Amount'].sum()
    average_transaction_amount = df['Amount'].mean()
    min_transaction_amount = df['Amount'].min()
    max_transaction_amount = df['Amount'].max()

    st.subheader('Summary Statistics')
    st.write(f"Total Transactions: {total_transactions}")
    st.write(f"Total Amount Spent: ₹{total_amount_spent:.2f}")
    st.write(f"Average Transaction Amount: ₹{average_transaction_amount:.2f}")
    st.write(f"Minimum Transaction Amount: ₹{min_transaction_amount}")
    st.write(f"Maximum Transaction Amount: ₹{max_transaction_amount}")

    st.subheader('Preview of DataFrame')
    if st.checkbox('Show DataFrame Preview'):
        st.write(df.head())

def transaction_type_analysis():
    st.subheader('Transaction Type Analysis')

    # Pie chart for transaction types
    transaction_type_counts = df['Transaction Type'].value_counts()
    st.write(transaction_type_counts)

    # Plotting a Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(transaction_type_counts, labels=transaction_type_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Transaction Type Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot()

def category_spending_analysis():
    st.subheader('Category-wise Spending Analysis')

    # Bar chart for category-wise spending
    category_spending = df.groupby('Transaction Type')['Amount'].sum()
    st.bar_chart(category_spending)

def receiver_analysis():
    st.subheader('Receiver-wise Analysis')

    # Horizontal bar chart for top receivers
    top_receivers = df.groupby('Receiver')['Amount'].sum().nlargest(10)
    st.bar_chart(top_receivers)

def date_analysis():
    st.subheader('Timeline Analysis')

    # Line chart for monthly spending trend
    monthly_spending = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
    st.line_chart(monthly_spending)

# Define the layout of your Streamlit app
st.title('Transaction Data Analysis')

# Sections for different types of analysis
overview_section()
transaction_type_analysis()
category_spending_analysis()
receiver_analysis()
date_analysis()
