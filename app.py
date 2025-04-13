import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio
import asyncio
import nest_asyncio
import json
from plotly.io import from_json

# Fix Streamlit event loop issue
nest_asyncio.apply()

# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# Set Plotly default template
pio.templates.default = "plotly_white"

st.set_page_config(page_title="Chatlytics: Business Data Insights", layout="wide")
st.title("📊 Chatlytics: Business Data Insights Chatbot")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "df" not in st.session_state:
    st.session_state.df = None
if "data_agent" not in st.session_state:
    st.session_state.data_agent = None
if "active_mode" not in st.session_state:  # Track active document type
    st.session_state.active_mode = None

def get_chart_config_llm_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a data visualization assistant. Based on the user's prompt and the dataset's columns, return a JSON with:
- chart_type: one of ["bar", "pie", "line", "scatter"]
- x_axis: (optional)
- y_axis: (optional)
- group_by: (optional)

Respond in JSON only. No explanation.

User prompt: {query}
Available columns: {columns}
""")
    return prompt | llm

def process_pdf(pdf_path):
    """Process PDF files for document QA"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(pages)
    
    # Updated embeddings initialization
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def process_data_file(file):
    """Process CSV/Excel files into DataFrame"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return None
        
        # Clean data using vectorized operations
        df = df.map(lambda x: x.encode('utf-8', 'ignore').decode('utf-8') 
                if isinstance(x, str) else x)
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

@tool
def generate_visualization(query: str) -> str:
    """
    Dynamically generate Plotly visualizations using LLM-based interpretation of user prompts.
    """
    try:
        df = st.session_state.df.copy()
        if df is None or df.empty:
            return "CHART|||NO_DATA|||ANALYSIS|||No data available."

        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        chain = get_chart_config_llm_chain(llm)
        result = chain.invoke({
            "query": query,
            "columns": ", ".join(df.columns)
        })

        from langchain.schema import AIMessage

        # Ensure it's a string
        if isinstance(result, AIMessage):
            result_text = result.content
        elif isinstance(result, str):
            result_text = result
        else:
            result_text = str(result)

        config = json.loads(result_text)

        chart_type = config.get("chart_type", "bar").lower()
        x = config.get("x_axis")
        y = config.get("y_axis")
        group_by = config.get("group_by")

        # st.write("📊 **DEBUG**: Chart Config from LLM =>", config)  # Debug output

        if group_by and group_by in df.columns:
            agg_df = df[group_by].value_counts().reset_index()
            agg_df.columns = [group_by, "Count"]
        elif x and x in df.columns:
            agg_df = df[x].value_counts().reset_index()
            agg_df.columns = [x, "Count"]
        else:
            return "CHART|||NO_DATA|||ANALYSIS|||Insufficient or invalid columns to generate chart."

        if chart_type == "pie":
            fig = px.pie(agg_df, names=agg_df.columns[0], values="Count", title=f"{agg_df.columns[0]} Distribution")
        elif chart_type == "line":
            fig = px.line(agg_df, x=agg_df.columns[0], y="Count", title=f"{agg_df.columns[0]} Trend")
        elif chart_type == "scatter":
            fig = px.scatter(agg_df, x=agg_df.columns[0], y="Count", title=f"{agg_df.columns[0]} Scatter")
        else:
            fig = px.bar(agg_df, x=agg_df.columns[0], y="Count", title=f"{agg_df.columns[0]} Bar Chart")

        return f"CHART|||{fig.to_json()}|||ANALYSIS|||Successfully generated a {chart_type} chart for '{agg_df.columns[0]}'."

    except Exception as e:
        # st.write("⚠️ **DEBUG**: Exception in generate_visualization =>", str(e))
        return f"CHART|||ERROR|||ANALYSIS|||Error generating chart: {str(e)}"

def create_dataframe_agent(df):
    """Create data analysis agent with visualization capability"""
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Revised prefix with a few-shot example
    prefix = """
You are a data analysis expert. Follow these rules:
1. ALWAYS use generate_visualization for charts
2. Never use matplotlib or python_repl_ast
3. Provide final answer in the format: CHART|||<chart JSON>|||ANALYSIS|||<analysis text>
4. Handle dates carefully

Below is an example of how you should respond:

EXAMPLE
-------
User: "Can you create a pie chart of Sales by Region?"
Assistant:
Thought: "I should use the generate_visualization tool to build the chart"
Action: generate_visualization
Action Input: "Pie chart of Sales by Region"

Observation: 
CHART|||{"data": [...], "layout": {...}}|||ANALYSIS|||Based on the pie chart, Region A leads in sales.

# Final Answer from the assistant:
CHART|||{"data": [...], "layout": {...}}|||ANALYSIS|||Based on the pie chart, Region A leads in sales...
-------
END OF EXAMPLE
"""

    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="openai-tools",
        max_iterations=5,
        extra_tools=[generate_visualization],
        allow_dangerous_code=True,
        prefix=prefix
    )


# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Files")

    pdf_file = st.file_uploader("PDF Document", type="pdf")
    data_file = st.file_uploader("Data File (CSV/Excel)", type=["csv", "xls", "xlsx"])

    # If both are uploaded, show a warning and stop execution
    if pdf_file and data_file:
        st.warning("Please upload only one file at a time! Remove one of them.")
        st.stop()

# If there's only a PDF and no CSV
if pdf_file:
    try:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        st.session_state.qa_chain = process_pdf("temp.pdf")
        st.session_state.active_mode = "pdf"
        # Clear data file context
        st.session_state.df = None
        st.session_state.data_agent = None
        st.session_state.current_data_file = None
        st.success("PDF document processed!")
    except Exception as e:
        st.error(f"PDF Error: {str(e)}")

# If there's only a CSV (and no PDF)
if data_file and data_file.name != st.session_state.get('current_data_file'):
    with st.spinner("Analyzing data file..."):
        df = process_data_file(data_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.data_agent = create_dataframe_agent(df)
            st.session_state.current_data_file = data_file.name
            st.session_state.active_mode = "data"
            # Clear PDF context
            st.session_state.qa_chain = None
            st.success("Data file processed!")

# Chat interface
if prompt := st.chat_input("Ask about your data or document"):
    # st.write("🔎 **DEBUG**: User propt =>", prompt)  # Debug statement

    # Check which mode is active
    if st.session_state.active_mode == "data" and st.session_state.data_agent and st.session_state.df is not None:
        try:
            response = st.session_state.data_agent.invoke({"input": prompt})
            
            # DEBUG: Show the raw response
            # st.write("🔎 **DEBUG**: Agent response =>", response)

            if isinstance(response, dict) and "output" in response:
                output_text = response["output"]
            elif isinstance(response, str):
                output_text = response
            else:
                output_text = str(response)

            # DEBUG: Show the final output text
            # st.write("🔎 **DEBUG**: Output text =>", output_text)

            with st.chat_message("assistant"):
                # Check if it contains CHART|||
                if "CHART|||" in output_text:
                    parts = output_text.split("|||")
                    if len(parts) >= 4:
                        chart_json = parts[1]  # "NO_DATA" or actual JSON
                        analysis_text = parts[3]
                        
                        if chart_json == "NO_DATA":
                            # No valid chart, but still show the "analysis_text"
                            st.markdown("**Analysis (No Chart):**")
                            st.write(analysis_text)
                        else:
                            # Attempt to load a real chart
                            try:
                                fig = from_json(chart_json)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error("⚠️ Could not render chart.")
                                st.code(chart_json, language="json")
                            
                            # Then show the LLM’s analysis
                            st.markdown("**Analysis:**")
                            st.write(analysis_text)
                    else:
                        st.warning("CHART message has unexpected format.")
                else:
                    # If "CHART|||" not in output_text at all, show the entire text
                    st.write(output_text)

                
                # Always show data sample
                st.write("**Data Sample:**")
                st.dataframe(st.session_state.df.sample(3))
                    
        except Exception as e:
            # st.write("⚠️ **DEBUG**: Exception in data block =>", str(e))
            st.error(f"Data Analysis Error: {str(e)}")
    
    elif st.session_state.active_mode == "pdf" and st.session_state.qa_chain:
        try:
            result = st.session_state.qa_chain({"query": prompt})
            with st.chat_message("assistant"):
                st.write(result["result"])
                with st.expander("Source Context"):
                    st.write(result["source_documents"][0].page_content)
        except Exception as e:
            # st.write("⚠️ **DEBUG**: Exception in pdf block =>", str(e))
            st.error(f"Document Query Error: {str(e)}")
    
    else:
        st.warning("Please upload a file first!")

if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY in .env file!")
