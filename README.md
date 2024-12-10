# AI Dashbot: Generative AI-Powered Sales Dashboard  

AI Dashbot is a dynamic, generative AI-powered dashboard designed to provide actionable insights into store performance. It features interactive visualizations, a chatbot powered by AI models like OpenAI and Gemini, and filtering options for tailored data analysis, supporting better decision-making.

---

## 🛠 Features  

- **Sales Dashboard**  
  Visualize and analyze store performance with interactive charts and insights.  

- **AI Chat Bot**  
  Engage with a chatbot that leverages advanced AI models for intelligent responses and data queries.  

- **Streamlined Navigation**  
  Easily toggle between the dashboard and chatbot with a toolbar-style navigation system.  

---

## 🚀 Live Demo  

Explore the live app: [AI Dashbot](https://aidashbot.streamlit.app/)  

---
📂 Project Structure
bash
Copy code
AI Dashbot/
│
├── .streamlit/             # Streamlit configuration files  
├── assets/                 # Static assets (e.g., logos, images)  
├── views/                  # Application views (dashboard and chatbot)  
│   ├── sales_dashboard.py  # Sales Dashboard logic  
│   └── chatbot.py          # AI Chat Bot logic  
├── Dataset.csv             # Sample dataset used for visualization  
├── app.py                  # Main application entry point  
├── requirements.txt        # Python dependencies  
└── .gitignore              # Files to ignore in version control  
⚙️ Installation
Clone the repository

bash
Copy code
git clone https://github.com/midhunpaul396/AIDashboardBot.git  
cd AIDashboardBot  
Set up a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv  
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`  
Install dependencies

bash
Copy code
pip install -r requirements.txt  
Run the application

bash
Copy code
streamlit run app.py  
📊 Dataset
The application uses a sample dataset (Dataset.csv) for generating insights. You can replace this with your own data to customize the analysis.

🤝 Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

🔗 Links
Live Preview: https://aidashbot.streamlit.app/
GitHub Repository: https://github.com/midhunpaul396/AIDashboardBot
