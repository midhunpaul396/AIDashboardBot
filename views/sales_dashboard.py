import google.generativeai as genai
import pandas as pd
import plotly.express as px
import streamlit as st
import openai
from fpdf import FPDF
import base64


# Fetch the OpenAI API key
# openai.api_key = st.secrets["OPENAI_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]  # Fetch Gemini API key

# Set Streamlit page configuration
# st.set_page_config(page_title="AI Sales Dashboard", page_icon=":bar_chart:", layout="wide")

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Sidebar filters
st.sidebar.header("Please Filter Here:")


# Check if filters are in session state; if not, initialize them
if "Region" not in st.session_state:
    st.session_state.Region = df["Region"].unique()
if "Segment" not in st.session_state:
    st.session_state.Segment = df["Segment"].unique()
if "Ship_Mode" not in st.session_state:
    st.session_state.Ship_Mode = df["Ship_Mode"].unique()

def regionCallback():
    st.session_state.Region = st.session_state.RegKey
def SegmentCallback():
    st.session_state.Segment = st.session_state.SegKey
def shipModeCallback():
    st.session_state.Ship_Mode = st.session_state.ShipKey


# Sidebar multiselect filters using session state

Region = st.sidebar.multiselect(
    "Select the Region:",
    options=df["Region"].unique(),
    default=st.session_state.Region,
    on_change=regionCallback,
    key = 'RegKey'
)
Segment = st.sidebar.multiselect(
    "Select the Segment:",
    options=df["Segment"].unique(),
    default=st.session_state.Segment,
    on_change=SegmentCallback,
    key = 'SegKey'

)
Ship_Mode = st.sidebar.multiselect(
    "Select the Ship Mode:",
    options=df["Ship_Mode"].unique(),
    default=st.session_state.Ship_Mode,
    on_change=shipModeCallback,
    key = 'ShipKey'
)

# Update session state with the latest filter values
st.session_state.Region = Region
st.session_state.Segment = Segment
st.session_state.Ship_Mode = Ship_Mode

# Data selection based on filters
df_selection = df.query(
    "Region == @Region & Segment == @Segment & Ship_Mode == @Ship_Mode"
)

if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

# ---- MAIN PAGE ----s
st.header(":bar_chart: AI Sales Dashboard")
st.markdown("### Key Metrics")

# Key Performance Indicators (KPIs)
total_sales = round(df_selection["Sales"].sum(), 2)  # Total Sales
total_profit = round(df_selection["Profit"].sum(), 2)  # Total Profit
average_discount = round(df_selection["Discount"].mean() * 100, 1)  # Average Discount in %
total_orders = df_selection["Order ID"].nunique()  # Total Unique Orders
average_sales_per_order = round(total_sales / total_orders, 2)  # Avg Sales/Order
total_quantity_sold = df_selection["Quantity"].sum()  # Total Quantity Sold
total_discount = round(df_selection["Discount"].sum(), 2)  # Total Discount
profit_margin = round((total_profit / total_sales) * 100, 2)  # Profit Margin
sales_per_customer = round(total_sales / df_selection['Customer ID'].nunique(), 2)  # Sales per Customer

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

# Displaying KPIs
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.markdown(f"### Total Sales\n**${total_sales:,}**")
with middle_column:
    st.markdown(f"### Total Profit\n**${total_profit:,}**")
with right_column:
    st.markdown(f"### Avg Sales/Order\n**${average_sales_per_order:,}**")

st.markdown("---")

# Sales and Profit by Product Line [Stacked Area Chart]
sales_profit_by_product_line = (
    df_selection.groupby(by=["Sub_Category"])[["Sales", "Profit"]]
    .sum()
    .sort_values(by="Sales", ascending=False)
)

fig_sales_profit_area = px.area(
    sales_profit_by_product_line,
    x=sales_profit_by_product_line.index,
    y=["Sales", "Profit"],
    title="Sales and Profit by Product Line",
    labels={"value": "Amount (USD)", "variable": "Metric"},
    template="plotly_white",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_sales_profit_area.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, title="Product Line"),
    yaxis=dict(title="Amount (USD)", showgrid=True),
)

st.plotly_chart(fig_sales_profit_area, use_container_width=True)

# Sales and Profit by Category [Bar Chart]
sales_profit_by_category = (
    df_selection.groupby("Category")[["Sales", "Profit"]]
    .sum()
    .reset_index()
)

fig_sales_profit = px.bar(
    sales_profit_by_category,
    x="Category",
    y=["Sales", "Profit"],
    title="Sales and Profit by Category",
    barmode="group",
    labels={"value": "Amount (USD)", "variable": "Metric"},
    template="plotly_white",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_sales_profit.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

st.plotly_chart(fig_sales_profit, use_container_width=True)

# Sales by Region [Pie Chart]
sales_profit_by_region = df_selection.groupby("Region")[["Sales", "Profit"]].sum().reset_index()

fig_sales_profit_region = px.pie(
    sales_profit_by_region,
    names="Region",
    values="Sales",
    title="Total Sales by Region",
    template="plotly_white"
)

fig_sales_profit_region.update_traces(textinfo="percent+label")
st.plotly_chart(fig_sales_profit_region, use_container_width=True)

# ---- AI-GENERATED INSIGHTS ----
st.markdown("### AI-Generated Insights")

# Enhanced prompt with more context
data_summary = (
    f"Dataset contains {len(df_selection)} records after filtering.\n"
    f"Total Sales: ${total_sales:,}\n"
    f"Total Profit: ${total_profit:,}\n"
    f"Average Discount: {average_discount}%\n"
    f"Average Sales per Order: ${average_sales_per_order:,}\n"
    f"Total Quantity Sold: {total_quantity_sold:,} units\n"
    f"Total Discount Given: ${total_discount:,}\n"
    f"Profit Margin: {profit_margin}%\n"
    f"Sales per Customer: ${sales_per_customer:,}\n"
    f"Number of Unique Customers: {df_selection['Customer ID'].nunique()}\n\n"
    f"Please analyze this data for potential patterns, identify anomalies or outliers, and provide actionable insights "
    f"that can help optimize sales and profits. Highlight any trends in sales by product category or region. "
    f"Consider analyzing the relationship between discount rates, sales, and profit margins for further opportunities "
    f"to optimize pricing strategies. Identify any trends related to customer purchasing behaviors, such as repeat buyers "
    f"or the impact of discounts on customer loyalty."
)

# Button to trigger insights generation
if st.button("Generate Insights"):
    with st.spinner("Generating insights..."):
        if True:  # Check if the OpenAI API key is missing
            if gemini_api_key:  # If Gemini API key exists, use Gemini
                try:
                    # Configure Gemini API
                    genai.configure(api_key=gemini_api_key)

                    generation_config = {
                        "temperature": 1,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                        "response_mime_type": "text/plain",
                    }

                    # Create model with Gemini
                    model = genai.GenerativeModel(
                        model_name="gemini-1.5-pro",
                        generation_config=generation_config,
                        system_instruction="You are an expert data analyst. Your goal is to provide insights from KPI's generated from a dataset's analysis.",
                    )

                    # Start chat session
                    chat_session = model.start_chat(
                        history=[
                            {"role": "user", "parts": [data_summary]},
                        ]
                    )

                    # Generate insights using Gemini
                    response = chat_session.send_message(data_summary)

                    st.session_state["gemini_insights"] = response.text

                    # Display Gemini insights
                    st.subheader("AI-Generated Insights")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating insights with Gemini: {e}")
            else:
                st.error("Sorry, I am out of API fuel for both OpenAI and Gemini.")
        else:
            try:
                # Get OpenAI insights if the API key exists
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are an expert data analyst."},
                              {"role": "user", "content": data_summary}],
                    max_tokens=600,
                    temperature=0.7,
                )

                # Extract and display OpenAI insights
                ai_insights = response["choices"][0]["message"]["content"].strip()
                st.subheader("AI-Generated Insights")
                st.write(ai_insights)

            except Exception as e:
                st.error(f"Error generating insights with OpenAI: {e}")
# Function to generate a PDF report
def generate_pdf_report(kpis, visualizations, gemini_insights):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="AI Sales Dashboard Report", ln=True, align="C")
    pdf.ln(10)

    # KPIs
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Key Metrics", ln=True)
    pdf.set_font("Arial", size=12)
    for kpi, value in kpis.items():
        pdf.cell(200, 10, txt=f"{kpi}: {value}", ln=True)

    pdf.ln(10)

    # AI Insights
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="AI-Generated Insights", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, gemini_insights)

    pdf.ln(10)

    # Visualizations
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Visualizations", ln=True)
    pdf.set_font("Arial", size=12)
    for viz_title, viz_path in visualizations.items():
        pdf.cell(200, 10, txt=viz_title, ln=True)
        pdf.image(viz_path, w=170)
        pdf.ln(10)

    return pdf

include_ai_insights = st.checkbox("Include AI Insights in the Report", value=False)
# Button to generate and download the report
if st.button("Download PDF Report"):
    with st.spinner("Generating PDF report..."):
        try:
            # Prepare KPIs, insights, and visualizations for the report
            kpis = {
                "Total Sales": f"${total_sales:,}",
                "Total Profit": f"${total_profit:,}",
                "Average Discount": f"{average_discount}%",
                "Total Orders": total_orders,
                "Average Sales/Order": f"${average_sales_per_order:,}",
                "Total Quantity Sold": total_quantity_sold,
                "Profit Margin": f"{profit_margin}%",
                "Sales per Customer": f"${sales_per_customer:,}"
            }

            # Example paths to images for visualizations
            sales_profit_by_product_line_path = "sales_profit_by_product_line.png"
            sales_profit_by_category_path = "sales_profit_by_category.png"
            sales_by_region_path = "sales_by_region.png"

            # Save visualizations to files
            fig_sales_profit_area.write_image(sales_profit_by_product_line_path)
            fig_sales_profit.write_image(sales_profit_by_category_path)
            fig_sales_profit_region.write_image(sales_by_region_path)

            visualizations = {
                "Sales and Profit by Product Line": sales_profit_by_product_line_path,
                "Sales and Profit by Category": sales_profit_by_category_path,
                "Total Sales by Region": sales_by_region_path
            }

            # Conditionally include AI insights
            gemini_insights = st.session_state.get("gemini_insights", "No Gemini insights available.") if include_ai_insights else ""

            # Generate PDFs
            pdf = generate_pdf_report(kpis, visualizations, gemini_insights)

            # Save PDF to a temporary file
            pdf_path = "AI_Sales_Dashboard_Report.pdf"
            pdf.output(pdf_path)

            # Encode PDF for download
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()

            b64 = base64.b64encode(pdf_data).decode()  # Base64 encoding
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="AI_Sales_Dashboard_Report.pdf">Click here to download the report</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating PDF report: {e}")
