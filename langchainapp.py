import streamlit as st
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv
import pandas as pd
import os
from langchain.output_parsers.fix import OutputFixingParser
from langchain.prompts import PromptTemplate

# Load environment variables
st.title("Data Analysis Agent with Streamlit")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

if not api_key:
    
    st.error("Please enter your OpenRouter API Key in the sidebar.")

else:
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("File loaded successfully! Here's a preview of your data:")
        st.dataframe(df.head(5))

        # Initialize the LLM
        llm = OpenAI(
            model_name="deepseek/deepseek-v3-base:free",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            streaming=True,
            temperature=0)
        
        # Set up prompt template and output parser
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="You are analyzing a pandas dataframe. Provide your answer in JSON format with keys 'thought', 'action', and 'result'. If your response includes code, ensure it is clearly marked as an 'action'. Question: {input}"
        )

        output_parser = OutputFixingParser.from_llm(
            llm=llm,
            parser=None,
            prompt=prompt_template,
            max_retries=2
        )

        # Create the agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            output_parser=output_parser, 
                        max_iterations=5,
            early_stopping_method="generate"

        )
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your data:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # Run agent without callback handler to suppress intermediate output
                    response = agent.invoke({"input": prompt})
                    
                    # Extract components
                    final_answer = response.get("output", "")
                    action_inputs = [
                        step[0].tool_input 
                        for step in response.get("intermediate_steps", []) 
                        if step and hasattr(step[0], 'tool_input')
                    ]
                    
                    # Format clean output
                    output = f"**Final Answer**: {final_answer}"
                    if action_inputs:
                        output += "\n\n**Code Used**:\n```python\n"
                        output += "\n".join(str(input) for input in action_inputs)
                        output += "\n```"
                    
                    st.markdown(output)

                except Exception as e:
                    output = f"Error: {str(e)}. Please try rephrasing your question."
                    st.markdown(output)

            st.session_state.messages.append({"role": "assistant", "content": output})
    else:
        st.info("Please upload a CSV file to get started.")
