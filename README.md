# 📊 Chatlytics: Business Data Insights Chatbot
InsightBot is an intelligent Streamlit-based chatbot that provides conversational insights from:

Uploaded PDF reports using RAG-based document QA
Uploaded CSV/Excel files using LLM-based data analysis and visualization
🚀 Features
Upload PDF files and ask questions — powered by LangChain + FAISS
Upload CSV/XLSX data files and ask analytical queries
Dynamic charts (bar, pie, line, scatter) powered by Plotly & LLM chart config generation
Uses ChatGroq and LLaMA3-70B for powerful natural language understanding
Clean Streamlit interface with automatic chart rendering & data samples
🗂️ Project Structure
CHATBOT/
├── venv/                 # Virtual environment
├── .env                  # Environment variables (GROQ_API_KEY required)
├── .gitignore
├── app.py                # Main application
└── requirements.txt      # Python dependencies
📦 Setup Instructions
Clone the repository
git clone <your_repo_url>
cd CHATBOT
Create a virtual environment & activate
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install dependencies
pip install -r requirements.txt
Set environment variables
Create a .env file and add your GROQ API key:
GROQ_API_KEY=your_groq_api_key
Run the Streamlit app
streamlit run app.py
🧪 Sample Prompts You Can Try
For PDF Reports:
"Summarize the key highlights from the report."
"What does the report say about sales in Asia?"
"Give me the top insights from the business summary."
For CSV/Excel Data:
"Create a bar chart of sales by country."
"Show a pie chart of region distribution."
"What is the average sales per region?"
"Which country has the highest sales?"
"Can you find trends in sales over time (if date column exists)?"
📤 Uploading & Interacting
Upload a PDF to ask questions about the document (retrieval-based answers)
Upload a CSV/Excel to ask questions and generate visual insights
🛠️ Technologies Used
Streamlit
LangChain, ChatGroq, FAISS
Plotly
Pandas, NumPy
dotenv
⚠️ Notes
Requires a valid GROQ_API_KEY in .env
Tested with LLaMA3-70B via Groq API
Made with ❤️ for business data lovers.
