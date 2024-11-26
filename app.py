import streamlit as st
import streamlit_nested_layout

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
import time

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

# Initialize ChatOpenAI for TogetherAI
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    streaming=True,
)


def ms_to_min_sec(ms):
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    return f"{int(minutes):02d}:{int(seconds):02d}"


def format_docs(docs):
    formatted_docs = "\n\n".join(
        [
            json.dumps({"content": doc.page_content, "metadata": doc.metadata})
            for doc in docs
        ]
    )

    citations = sorted(
        [
            {
                "title": f"Lecture {int(doc.metadata['lecture_number'])} ({ms_to_min_sec(doc.metadata.get('start_ms', 0))} - {ms_to_min_sec(doc.metadata.get('end_ms', 0))})",
                "content": doc.page_content,
                "lecture_number": int(doc.metadata["lecture_number"]),
                "start_ms": doc.metadata.get("start_ms", 0),
            }
            for doc in docs
        ],
        key=lambda x: (x["lecture_number"], x["start_ms"]),
    )

    return formatted_docs, citations


def retrieve_context(user_query):
    template = """
    You are a bot that extracts metadata filters from user queries. You are given a user query and you need to extract the metadata filters from the query. 
    This is useful for retrieving specific chunks of text from a document based on user queries. If you are not able to extract the metadata filters from the user query, you should return an empty dictionary.
    The following schema is used to represent the metadata filters:
        {{
            "document_type": str, # one of "chapter summary", "transcript"
            "lecture_number": int, # the lecture number that the chunk of text comes from
            "start_ms": int, # the start time of the chunk of text in milliseconds
            "end_ms": int # the end time of the chunk of text in milliseconds
        }}
    
    For example, given the following user query: "Summarize the first 10 minutes of lecture 1", your response should be:
    {{
      "lecture_number": 1,
      "start_ms": {{"$lte": 600000}},
    }}

    Another example, given the following user query: "Summarize the first half of lecture 1", your response should be:
    {{
      "lecture_number": 1,
      "start_ms": {{"$lte": 2700000}},
    }}

    Another example, given the following user: "What is the summary of the first chapter of lecture 2?", your response should be:
    {{
      "lecture_number": 2
    }}

    If you are not able to extract the metadata filters from the user query or are not fully confident in your extraction, your response should be:
    {{}}

    Extract the metadata filters from the user query below and respond only in JSON format. Do not include any additional text or explanations.
    User query: {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({"user_query": user_query})
    cleaned_response = response.content.replace("```", "").replace("\\n", "")

    try:
        metadata_filters = json.loads(cleaned_response)
    except json.JSONDecodeError:
        metadata_filters = {}

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 20, "filter": metadata_filters}
    )
    retrieved_context = retriever.invoke(user_query)
    formatted_docs, citations = format_docs(retrieved_context)
    return formatted_docs, citations, retrieved_context


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

    formatted_docs, citations, retrieved_context = retrieve_context(user_query)

    # Combine prompt and TogetherAI in LangChain pipeline
    chain = prompt | llm | StrOutputParser()

    # Stream response
    response_stream = chain.stream(
        {
            "retrieved_context": retrieved_context,
            "user_question": user_query,
            "chat_history": "\n\n".join([msg.content for msg in chat_history]),
        }
    )

    response = ""
    for chunk in response_stream:
        response += chunk
        yield response, citations


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
        with st.spinner("Generating response..."):
            response_generator = stream_response(
                user_query, st.session_state.chat_history
            )
            response_placeholder = st.empty()
            response, citations = "", []
            for response, citations in response_generator:
                response_placeholder.write(response)
                time.sleep(0.1)  # Simulate delay for streaming

        with st.expander("View Citations"):
            st.write("Here are the citations for the retrieved context:")
            for citation in citations:
                with st.expander(citation["title"]):
                    st.write(citation["content"])

    st.session_state.chat_history.append(AIMessage(content=response))
