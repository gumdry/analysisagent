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
openrouter_api_key = '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'

# Streamlit app setup
st.title("Data Analysis Agent with Streamlit")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password", value=openrouter_api_key)

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
            model_name="google/gemini-2.5-pro-exp-03-25:free",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            streaming=True
        )

        # Set up prompt template and output parser
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="You are a helpful a data analyst. Answer the following question as accurately as possible:\n\n{input}"
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
            output_parser=output_parser
        )

        # Chat interface
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
                callback = StreamlitCallbackHandler(st.container())
                response = agent.invoke(
                    {"input": prompt},
                    config={"callbacks": [callback], "handle_parsing_errors": True}
                )
                st.markdown(response["output"])

            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
    else:
        st.info("Please upload a CSV file to get started.")