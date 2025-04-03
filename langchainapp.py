import streamlit as st
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StdOutCallbackHandler
import pandas as pd
import os
from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
openrouter_api_key= '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'
# data_loc='/content/sample_data/california_housing_train.csv'

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("AI Data Analysis Agent")
st.markdown("""
This application allows you to upload a dataset and ask questions about it in natural language.
The AI agent will translate your questions into code, execute it, and return the results.
""")

# Sidebar for API key input and configuration
st.sidebar.header("Configuration")
openrouter_api_key = st.sidebar.text_input("Enter your OpenRouter API Key", type="password")
st.sidebar.markdown("This app uses Google's Gemini 2.5 model through OpenRouter.")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# Handle file upload
if uploaded_file is not None and not st.session_state.file_uploaded:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.file_uploaded = True
        
        # Display success message
        st.success("File loaded successfully!")
        
        # Initialize the LLM with OpenRouter if API key is provided
        if openrouter_api_key:
            llm = OpenAI(
                model_name="google/gemini-2.5-pro-exp-03-25:free",
                openai_api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                streaming=True
            )
            
            # Create the pandas dataframe agent
            st.session_state.agent = create_pandas_dataframe_agent(
                llm,
                st.session_state.df,
                verbose=False,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )
            st.success("AI Agent initialized and ready to answer questions!")
        else:
            st.warning("Please enter your OpenRouter API key to initialize the AI agent.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Display data preview if data is loaded
if st.session_state.df is not None:
    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head(5))
    
    # Display data statistics
    st.subheader("Data Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {st.session_state.df.shape[0]}")
        st.write(f"Columns: {st.session_state.df.shape[1]}")
    with col2:
        st.write(f"Data types: {', '.join(st.session_state.df.dtypes.astype(str).unique())}")
    
    # Asking questions interface
    st.subheader("Ask Questions About Your Data")
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and question and st.session_state.agent:
        try:
            with st.spinner("Analyzing your data..."):
                # Use a container to display the streaming output
                response_container = st.empty()
                
                # Define a custom callback to update the response in real-time
                class StreamlitCallbackHandler(StdOutCallbackHandler):
                    def __init__(self, container):
                        super().__init__()
                        self.container = container
                        self.output = ""
                    
                    def on_llm_new_token(self, token, **kwargs):
                        self.output += token
                        self.container.markdown(self.output)
                
                # Create the callback handler
                callback = StreamlitCallbackHandler(response_container)
                
                # Invoke the agent with the callback
                response = st.session_state.agent.invoke(
                    {"input": question},
                    config={"callbacks": [callback], "handle_parsing_errors": True}
                )
                
                # Display the final response
                st.subheader("Answer")
                st.write(response["output"])
        except Exception as e:
            st.error(f"Error processing question: {e}")
    elif not st.session_state.agent and question:
        st.warning("Please upload a file and provide an API key to initialize the AI agent.")

# Add information and help section
st.sidebar.subheader("Help")
st.sidebar.markdown("""
### Example Questions:
- What is the total number of women in the data?
- What's the average age grouped by gender?
- Create a histogram of salary distribution.
- Find correlations between all numeric columns.
- Which products have the highest sales?
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Langchain and Google's Gemini 2.5")
