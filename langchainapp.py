import streamlit as st
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv
import pandas as pd
import os
from langchain.output_parsers.fix import OutputFixingParser
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="resources/logo.jpg",
    layout="wide"
    )

st.logo("resources/logo.jpg", size="large")

# Available free models on OpenRouter
FREE_MODELS = {
    "DeepSeek Free": "deepseek/deepseek-v3-base:free",
    "Llama 4 Maverick": "meta-llama/llama-4-maverick:free",
    "Qwen 2.5 VL Instruct": "qwen/qwen2.5-vl-3b-instruct:free",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro-exp-03-25:free",
    "Qwerky 72B": "featherless/qwerky-72b:free",
    "Mistral Small 3.1": "mistralai/mistral-small-3.1-24b-instruct:free",
    "OlympicCoder 32B": "open-r1/olympiccoder-32b:free"
}
# App title and description
st.title("AI Data Analysis Agent")
st.markdown("""
This application allows you to upload a dataset and ask questions about it in natural language.
The AI agent will translate your questions into code, execute it, and return the results.
""")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    options=list(FREE_MODELS.keys()),
    index=0  # Default to DeepSeek
)


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
            model_name=FREE_MODELS[selected_model],  # Use selected model
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
            verbose=False,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            output_parser=output_parser, 
            max_iterations=5,
            early_stopping_method="generate", 
            return_intermediate_steps=True
        )

        # Initialize session state for chat messages
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
                st_callback = StreamlitCallbackHandler(st.container())
                try:
                    response = agent.invoke(
                        {"input": prompt},
                        config={"callbacks": [st_callback]}                    
                        )                    
                    # Extract components from response
                    final_answer = response["output"]
                    intermediate_steps = response["intermediate_steps"]


                    # Extract action inputs from intermediate steps
                    action_inputs = [  
                        str(step[0].tool_input) 
                        for step in intermediate_steps 
                        if step and hasattr(step[0], 'tool_input')
                    ]                    

                    # Format the output
                    formatted_output = f"**Final Answer**: {final_answer}"
                    if action_inputs:
                        formatted_output += "\n\n**Action Inputs**:\n" + "\n".join(
                            [f"- {input}" for input in action_inputs]
                        )                    

                    st.markdown(formatted_output)
                    output = formatted_output
                except Exception as e:
                    output = f"Error: {str(e)}. Please try rephrasing your question."
                    st.markdown(output)

            st.session_state.messages.append({"role": "assistant", "content": output})
    else:
        st.info("Please upload a CSV file to get started.")

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
