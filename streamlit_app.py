import pickle
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

def get_source_list(result):
  out = ''
  i = 1
  for doc in result['source_documents']:
    row = 'Source {}: {}\n'.format(i, doc.metadata['source'])
    out += row
    i+=1
  return out

###########################
# LLM Instantiation
###########################

# open a file, where you stored the pickled data
file = open('SEC_APA_faiss_vectorstore_V1', 'rb') 

# dump information to that file
docsearch1 = pickle.load(file)

# close the file
file.close()

KEY = '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'

system_template="""The following is a set of text extracted from legal documents dealing with the purchase, sale, or redistribution of company assets or equity.
You are a helpful bot that answers questions coming from lawyers related to the following knowledgebase of text.

Take note of the company tickers, URLs and accessation numbers that the texts below come from. Please cite them at the end of your answer.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


#memory = ConversationBufferMemory(llm=llm, input_key='question', output_key='answer', memory_key="chat_history", return_messages=True)
chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=KEY)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch1.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


###########################
# Streamlit App UI
###########################
import streamlit as st
import random
import time

st.title("Asset Purchase Agreement LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help? "):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = qa_chain(prompt)  #qa_chain({"query": prompt})
        
        assistant_response = result['answer'] #result["result"]
        assistant_response = assistant_response +  '\n\n ____________________________________ \n\n' + '\n\n FULL LIST OF SOURCES CONSIDERED: \n\n' + get_source_list(result)
        assustant_response = assistant_response.replace('\n', '\s\s\n') # this is because st.chat recognizes markdown, which requires two spaces before a newline character.
        # Simulate stream of response with milliseconds delay
        #for chunk in assistant_response.split():
        #    full_response += chunk + " "
        #    time.sleep(0.02)
        #    # Add a blinking cursor to simulate typing
        #    message_placeholder.markdown(full_response + "▌")
        #message_placeholder.markdown(full_response)
        message_placeholder.markdown(assistant_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})









