import streamlit as st
from azure.cosmos import CosmosClient
import logging
from topic_modelling import extract_topics_from_text
from cloud_config import *

# Cosmos DB Credentials
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize Cosmos DB Client
client = CosmosClient(ENDPOINT, KEY)
database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

# Function to fetch data from Cosmos DB
def fetch_chat_titles(limit=50):
    query = "SELECT c.ChatTitle FROM c ORDER BY c.TimeStamp DESC OFFSET 0 LIMIT @limit"
    params = [{"name": "@limit", "value": limit}]
    items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
    return [item["ChatTitle"] for item in items]

# Automatically fetch and analyze topics on startup if not done already
if "text_content" not in st.session_state or "topics" not in st.session_state:
    st.session_state["text_content"] = ""
    st.session_state["topics"] = ""

    # Fetch and analyze topics
    chat_titles = fetch_chat_titles()
    if chat_titles:
        text_content = " ".join(chat_titles)
        topics = extract_topics_from_text(text_content)
        
        # Store in session state
        st.session_state["text_content"] = text_content
        st.session_state["topics"] = topics

        # Display the topics
        st.subheader("Top Discussed Topics:")
        st.write(topics)
    else:
        st.warning("No chat titles found.")

# Streamlit UI
st.title("DBminer")

# User input for questions
if prompt := st.chat_input("Ask a question about your emails"):
    text_content = st.session_state["text_content"]
    topics = st.session_state["topics"]

    if text_content and topics:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            # Pass the relevant data to the model in the prompt
            response_stream = llmclient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who answers questions based on data from the database."},
                    {"role": "user", "content": f"Answer the user question based on the following data from the database:\n\nText Content: {text_content}\n\nHighlighted Topics: {topics}\n\nQuestion: {prompt}"}
                ],
                temperature=0.5,
                stream=True,
            )

        bot_response = ""
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            for chunk in response_stream:
                if chunk.choices:
                    bot_response += chunk.choices[0].delta.content or ""
                    response_placeholder.markdown(bot_response)

        st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    else:
        st.warning("Please fetch and analyze topics first.")
