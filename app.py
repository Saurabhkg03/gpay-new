import streamlit as st
import pandas as pd
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

    # Display the processed data as a Streamlit table
    st.table(df)
    
    st.success('Data processed and displayed as a Streamlit table')
