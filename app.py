import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_together import TogetherEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain


from pinecone.grpc import PineconeGRPC as Pinecone

from dotenv import load_dotenv
import os
import json

from pinecone_utils import load_index

# Load environment variables
load_dotenv()

# Setup API Keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# App Configuration
title = "CS224V LectureBot 📚"
st.set_page_config(page_title=title, page_icon="🤖")
st.title(title)

# Configure Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cs224v-lecturebot"
index = load_index(pc, index_name=index_name)

# initialize PineconeVectorStore and embeddings
together_embedding = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval"
)
vectorstore = PineconeVectorStore(index, embedding=together_embedding, text_key="text")


def format_docs(docs):
    print(
        "\n\n".join(
            [
                json.dumps({"content": doc.page_content, "metadata": doc.metadata})
                for doc in docs
            ]
        )
    )
    return "\n\n".join(
        [
            json.dumps({"content": doc.page_content, "metadata": doc.metadata})
            for doc in docs
        ]
    )


# Function to Stream Responses
def stream_response(user_query, chat_history):
    """
    Streams response from TogetherAI using LangChain's ChatOpenAI wrapper.
    """
    template = """
    Use the following pieces of context and chat history to answer the question at the end, and respond kindly.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise.
    Always say "thanks for asking!" at the end of the answer.

    Chat history: {chat_history}

    User question: {user_question}

    Retrieved context: {retrieved_context}
    """

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize ChatOpenAI for TogetherAI
    llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=TOGETHER_API_KEY,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        streaming=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 20}
    )
    retrieved_context = retriever.invoke(user_query)
    retrieved_context = format_docs(retrieved_context)

    # Combine prompt and TogetherAI in LangChain pipeline
    chain = prompt | llm | StrOutputParser()

    # Stream response
    return chain.stream(
        {
            "retrieved_context": retrieved_context,
            "user_question": user_query,
            "chat_history": chat_history,
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

    # Stream the LLM Response
    with st.chat_message("AI"):
        response_stream = stream_response(user_query, st.session_state.chat_history)
        response = st.write_stream(response_stream)

    st.session_state.chat_history.append(AIMessage(content=response))
