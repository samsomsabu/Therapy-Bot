# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
# import chainlit as cl

# DB_FAISS_PATH = 'vectorstore/db_faiss'

# custom_prompt_template = """Please provide a supportive and therapeutic response to the user's question. Remember to maintain a positive and empathetic tone.

# Context: {context} Question: {question}
6
# Helpful answer:
# """

# def set_custom_prompt():
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])  
#     return prompt

# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=False, chain_type_kwargs={'prompt': prompt})
#     return qa_chain

# def load_llm():
#     # Replace 'path/to/your/local/model' with the actual path to your local model file
#     local_model_path = r'C:\Users\Udhay12345\Downloads\Llama2-Medical-Chatbot-main\neural-chat-7b-v3-1.Q6_K.gguf'

#     llm = CTransformers(
#         model=local_model_path,
#         model_type="mistral",  # Update this if your model type is different
#         lib="avx2",  # Use 'avx2' for CPU, change if using a different architecture
#         max_new_tokens=512,
#         temperature=0.5
#     )
#     return llm

# def qa_bot():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)
#     return qa

# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({'query': query})
#     return response

# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to the Therapy Bot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     if chain is None:
#         return
#     # Assuming 'chain' has an asynchronous method 'acall' to handle the query
#     res = await chain.acall({'query': message.content})
#     answer = res['result']

#     await cl.Message(content=answer).send() 

#model.py

import smtplib
from email.mime.text import MIMEText
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """you're an expert therapist,Please provide a supportive and therapeutic response to the user's question.
if the question is not related to psychological or Therapeutic pkease say these qustions are out of context.

Context: {context} Question: {question}

Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])  
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=False, chain_type_kwargs={'prompt': prompt})
    return qa_chain

def load_llm():
    # Replace 'path/to/your/local/model' with the actual path to your local model file
    local_model_path = r'C:\Users\Udhay12345\Downloads\Llama2-Medical-Chatbot-main\neural-chat-7b-v3-1.Q6_K.gguf'

    llm = CTransformers(
        model=local_model_path,
        model_type="mistral",  # Update this if your model type is different
        lib="avx2",  # Use 'avx2' for CPU, change if using a different architecture
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

def send_notification(email, message):
    # SMTP server settings
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'badrisrp3836@gmail.com'
    smtp_password = 'hngb nzfa prsd adcy'

    # Email content
    subject = 'Suicidal Attempt Detected'
    body = f"Conversation related to suicidal attempts:\n\n{message}"

    # Create MIMEText object
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'therapybot@example.com'
    msg['To'] = email

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(msg['From'], msg['To'], msg.as_string())

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Therapy Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

# List of keywords related to suicidal conversation
suicidal_keywords = ['suicide', 'self-harm', 'end my life', 'suicidal thoughts', 'kill myself']

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        return

    # Check if the message content contains any suicidal keywords
    for keyword in suicidal_keywords:
        if keyword in message.content.lower():
            # Notify the specified email
            email = 'badrisrp3836@gmail.com'  # Specify the email address to notify
            send_notification(email, message.content)
            break  # Exit the loop once a keyword is found

    # Process the message with the therapy bot
    res = await chain.acall({'query': message.content})
    answer = res['result']

    await cl.Message(content=answer).send()