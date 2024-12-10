import openai
import pandas as pd
import streamlit as st
import google.generativeai as genai

# openai.api_key = st.secrets["OPENAI_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Streamlit UI setup
st.title("Chatbot")

# Initialize session state variables if not already initialized
if "total_sales" not in st.session_state:
    df = pd.read_csv('Dataset.csv')
    # Key Performance Indicators (KPIs)
    total_sales = round(df["Sales"].sum(), 2)  # Total Sales
    total_profit = round(df["Profit"].sum(), 2)  # Total Profit
    average_discount = round(df["Discount"].mean() * 100, 1)  # Average Discount in %
    total_orders = df["Order ID"].nunique()  # Total Unique Orders
    average_sales_per_order = round(total_sales / total_orders, 2)  # Avg Sales/Order
    total_quantity_sold = df["Quantity"].sum()  # Total Quantity Sold
    total_discount = round(df["Discount"].sum(), 2)  # Total Discount
    profit_margin = round((total_profit / total_sales) * 100, 2)  # Profit Margin
    sales_per_customer = round(total_sales / df['Customer ID'].nunique(), 2)  # Sales per Customer

    # Store KPIs in session state
    st.session_state.total_sales = total_sales
    st.session_state.total_profit = total_profit
    st.session_state.average_discount = average_discount
    st.session_state.total_orders = total_orders
    st.session_state.average_sales_per_order = average_sales_per_order
    st.session_state.total_quantity_sold = total_quantity_sold
    st.session_state.total_discount = total_discount
    st.session_state.profit_margin = profit_margin
    st.session_state.sales_per_customer = sales_per_customer

# Define the data summary context
data_summary = (
    f"Filtered Data Summary:\n"
    f"Total Sales: ${st.session_state.total_sales:,}\n"
    f"Total Profit: ${st.session_state.total_profit:,}\n"
    f"Average Discount: {st.session_state.average_discount}%\n"
    f"Average Sales per Order: ${st.session_state.average_sales_per_order:,}\n"
    f"Total Quantity Sold: {st.session_state.total_quantity_sold:,} units\n"
    f"Total Discount Given: ${st.session_state.total_discount:,}\n"
    f"Profit Margin: {st.session_state.profit_margin}%\n"
    f"Sales per Customer: ${st.session_state.sales_per_customer:,}\n"
    f"Total Unique Orders: {st.session_state.total_orders}\n\n"
    "You are an intelligent assistant that provides explanations about the key performance indicators (KPIs) and trends in sales and profit. "
    "You can also identify patterns, insights, and suggestions based on the given metrics and user queries. "
    "Ask me about sales trends, profit margins, customer behavior, product category performance, or anything else related to the dataset. "
    "I can help you understand which areas are performing well and where improvements can be made."
) 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the sales data:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response using OpenAI or Gemini
    try:
        with st.chat_message("assistant"):
            if True:
                # Create the model with Gemini's configuration
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 550,
                    "response_mime_type": "text/plain",
                }

                model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    generation_config=generation_config,
                    system_instruction="You are an intelligent assistant that provides explanations about the key performance indicators (KPIs) and trends in sales and profit.",
                )

                # Start chat with Gemini
                chat_session = model.start_chat(
                    history=[
                        {
                            "role": "user",
                            "parts": [
                                data_summary + "\n"
                            ],
                        },
                        {
                            "role": "user",
                            "parts": [
                                "Also, consider this prompt: " + prompt + "\n"
                            ],
                        },
                    ]
                )

                # Get response from Gemini (Ensure content is non-empty)
                assistant_response = chat_session.send_message("Reply to the prompt")

                # Extract the text from the response object
                assistant_response_text = assistant_response.text.strip()
                st.markdown(assistant_response_text)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response_text})

            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[ 
                        {"role": "system", "content": "You are an intelligent assistant that provides explanations about the key performance indicators (KPIs) and trends in sales and profit."},
                        {"role": "user", "content": data_summary},  # Include the data summary as context
                        {"role": "user", "content": prompt},  # User's input query
                    ],
                    max_tokens=550,
                    temperature=0.7,
                )

                # Extract and display assistant response
                assistant_response = response["choices"][0]["message"]["content"].strip()

                # Display the assistant's response
                st.markdown(assistant_response.strip())

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response.strip()})

    except Exception as e:
        st.error(f"Error generating response: {e}")
