import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set TogetherAI API key
# Replace 'userdata.get' with 'os.getenv' or similar if 'userdata' is not defined in your project
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

# App Configuration
title = "CS224V LectureBot 📚"
st.set_page_config(page_title=title, page_icon="🤖")
st.title(title)


# Function to Stream Responses
def stream_response(user_query, chat_history):
    """
    Streams response from TogetherAI using LangChain's ChatOpenAI wrapper.
    """
    template = """
    Use the following pieces of context to answer the question at the end, and respond kindly.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    Chat history: {chat_history}

    User question: {user_question}
    """

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize ChatOpenAI for TogetherAI
    llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        streaming=True,
    )

    # Combine prompt and TogetherAI in LangChain pipeline
    chain = prompt | llm | StrOutputParser()

    # Stream response
    return chain.stream(
        {
            "chat_history": chat_history,
            "user_question": user_query,
        }
    )


# Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I am the CS224V LectureBot. How can I assist you today?")
    ]

# Display Chat History
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle User Input
user_query = st.chat_input("Type your question here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # TODO: do RAG stuff here

    # Stream the LLM Response
    # TODO: input context
    with st.chat_message("AI"):
        response_stream = stream_response(user_query, st.session_state.chat_history)
        response = st.write_stream(response_stream)

    st.session_state.chat_history.append(AIMessage(content=response))
