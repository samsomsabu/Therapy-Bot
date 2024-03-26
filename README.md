# Suicide Prevention: A Therapy ChatBot
### Developed By: Sundharess B, BadriNarayanan S, Samson Sabu, Tanisha Das
### Video Link - https://drive.google.com/file/d/1ruLFDL-BybDq0fnyfCaOfgf35QI35Qr1/view?usp=sharing

## Overview

Suicide Prevention: A Therapy ChatBot is an innovative therapy chatbot designed to provide immediate, empathetic support for individuals experiencing emotional distress or suicidal thoughts. By leveraging the power of the Intel model `TheBloke/neural-chat-7B-v3-1-GGUF`, Lifeline Harmony offers a conversational experience that understands and responds to the unique needs of each user.

## Purpose

The chatbot serves as a first line of defense in suicide prevention, offering a safe space for users to express their feelings and receive guidance. Its primary goal is to reduce the risk of suicide by providing support and directing users to appropriate resources.

## Capabilities

- **Emotional Support**: Engages users in therapeutic conversations to explore their feelings and provide comfort.
- **Crisis Detection**: Utilizes advanced algorithms to detect signs of suicidal ideation and trigger alerts.
- **Immediate Assistance**: Offers strategies and techniques to help users cope with their immediate emotional state.
- **Resource Connection**: Guides users to professional help and emergency services when necessary.

## Impact

Suicide Prevention: A Therapy ChatBot aims to make mental health support more accessible and immediate, potentially saving lives by connecting users with the help they need when they need it most.

For more information on how Suicide Prevention: A Therapy ChatBot is transforming mental health support, please refer to the detailed sections within this README.

# Key Packages

1) langchain - Seems to be a custom or specialized library, given the multiple modules imported from it (like vectorstores, embeddings, llms, and chains).
2) transformers - A library by Hugging Face that provides pre-trained models for Natural Language Processing (NLP) tasks like text classification, translation, and more. It includes models like BERT, GPT-2, T5, and others.
3) chainlit - Chainlit is an open-source asynchronous Python framework that enables developers to build scalable Conversational AI or agentic applications quickly and efficiently.

# Setup

### Clone the repository


### Install required dependencies

pip install -r requirements.txt

### Run the ingestion script to create the vector database

py ingest.py

### Start the ChatBot

python -m chainlit run model.py

# Requirements

To run this project, you will need the following packages:

- `smtplib`: The standard Python library for sending emails using the Simple Mail Transfer Protocol (SMTP).
- `email`: A package for managing email messages, including `MIMEText` for creating MIME-formatted text emails.
- `langchain_community.document_loaders`: Provides `PyPDFLoader` and `DirectoryLoader` for loading documents into the application.
- `langchain.prompts`: Includes `PromptTemplate` for creating custom prompt templates.
- `langchain_community.embeddings`: Contains `HuggingFaceEmbeddings` for utilizing embedding models from Hugging Face.
- `langchain_community.vectorstores`: Includes `FAISS` for efficient similarity search and clustering of dense vectors.
- `langchain_community.llms`: Contains `CTransformers` for working with transformer models.
- `langchain.chains`: Includes `RetrievalQA` for building retrieval-based question-answering systems.
- `chainlit`: A package for running LangChain applications with ChainLit.

Make sure to install these packages using `pip` before running the project.

# Workflow

This section outlines the workflow of our therapy chatbot, designed to provide support and prevent suicidal attempts. The workflow is divided into several key stages:

### User Interaction
- Users begin by interacting with the chatbot through a conversational interface.
- The chatbot utilizes natural language processing to understand and respond to user inputs.

![Screenshot (117)](https://github.com/SundharessB/Therapy-ChatBot/assets/139948283/ad713001-019d-4442-b5e9-888cfc16df5d)

### Assistance and Support
- The chatbot provides empathetic responses and engages users in therapeutic conversations.
- It offers guidance and support based on the user's expressed emotions and concerns.

![Screenshot (118)](https://github.com/SundharessB/Therapy-ChatBot/assets/139948283/482ce68a-d63f-4fe6-aedf-b84e7e24528d)

### Alert Mechanism

- Upon detecting potential suicidal intent, the chatbot triggers an automated alert system.
- An email notification is sent to a designated organization for immediate human intervention.

![Screenshot (119)](https://github.com/SundharessB/Therapy-ChatBot/assets/139948283/99e59f9c-6a4e-4614-a8b7-7860bcd42b78)


### Follow-Up

- The organization can then reach out to the user to provide further assistance and resources.
- The chatbot also provides information on how to access professional help and emergency services.

### Continuous Improvement

- User interactions are analyzed to improve the chatbot's responses and detection capabilities.
- Feedback loops are established to refine the system and enhance its effectiveness.

# Role of Vector Database

A **Vector Database** is a specialized database designed to store, manage, and index high-dimensional vector data efficiently. These vectors, also known as embeddings, are numerical representations of data objects that carry semantic information.

In our therapy chatbot, the vector database plays a crucial role in understanding and processing user inputs. By converting text into vector embeddings, the chatbot can analyze the conversation's context, detect crisis situations, and provide relevant and personalized support to the user.

The use of a vector database ensures that our chatbot responds with high accuracy and relevance, making it a vital component in our mission to offer timely assistance and prevent suicidal attempts.

This workflow ensures that users receive timely and appropriate support, while also facilitating human intervention when necessary.

# Intel Products Utilized in Chatbot Development

Our therapy chatbot leverages several Intel products that enhance its capabilities and performance. Below is a list of the key Intel technologies integrated into our chatbot framework:

### Intel® Extension for Transformers

NeuralChat comes under this extension.

### NeuralChat

NeuralChat is a customizable chatbot framework under the Intel® Extension for Transformers. It allows for the creation of chatbots with a rich set of plugins for knowledge retrieval, interactive speech, query caching, and security guardrails. NeuralChat is built on top of large language models (LLMs) and is designed to be deployed easily as a service.

### Intel® Developer Cloud

The Intel Developer Cloud provides access to various hardware and software resources, enabling developers to build and test AI applications like chatbots. It simplifies the deployment process and offers optimizations using Intel AI software.

These Intel products provide the necessary infrastructure and tools to create a responsive and efficient therapy chatbot capable of offering real-time support and crisis intervention.

# Acknowledgement

We would like to extend our heartfelt gratitude to the Intel oneAPI community for their invaluable contributions to the development of our therapy chatbot, **Calm Companion**. The resources, tools, and support provided by the oneAPI initiative have been instrumental in enabling us to leverage cutting-edge technology to make a positive impact on mental health. This project stands as a testament to what can be achieved when we come together to address critical challenges in our society.

Thank you, Intel oneAPI community, for your support and for helping us bring **Calm Companion** to life.












