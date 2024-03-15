import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from datasets import load_dataset
import pinecone
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar
from langchain.vectorstores import Pinecone
import os
from PyPDF2 import PdfReader
import re


load_dotenv()


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


# PDF file path
pdf_path = "../pdfs/portal-be-runbook.pdf"
absolute_path = os.path.abspath(pdf_path)
print(absolute_path)

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)


# Function to preprocess text
def preprocess_text(text):
    # Remove special characters, multiple spaces, and newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.strip()  # Remove leading and trailing spaces
    return text


# Preprocess the extracted text
pdf_text = preprocess_text(pdf_text)
# print(pdf_text)



# ============================================
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

messages = []

#
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Hi AI, how are you today?"),
#     AIMessage(content="I'm great thank you. How can I help you?"),
#     HumanMessage(content="I'd like to understand string theory.")
# ]
#
# res = chat(messages)
# # print(res.content)
#
# messages.append(res)
#
# prompt = HumanMessage(
#     content="Why do physicists believe it can produce a 'unified theory'?"
# )
#
# messages.append(prompt)
#
# # send to chat-gpt (gpt-3.5-turbo)
# res = chat(messages)
#
# # print(res.content)
#
# messages.append(res)
#
# prompt = HumanMessage(
#     content="What is so special about Llama 2?"
# )
#
# messages.append(prompt)
#
# # send to chat-gpt (gpt-3.5-turbo)
# res = chat(messages)
#
# # print(res.content)
#
# messages.append(res)
#
# prompt = HumanMessage(
#     content="Can you tell me about the LLMChain in LangChain?"
# )
#
# messages.append(prompt)
#
# res = chat(messages)
#
# # print(res.content)
#
# llmchain_information = [
#     "A LLMChain is the most common type of chain. It consists of a PromptTemplate, a model (either an LLM or a ChatModel), and an optional output parser. This chain takes multiple input variables, uses the PromptTemplate to format them into a prompt. It then passes that to the model. Finally, it uses the OutputParser (if provided) to parse the output of the LLM into a final format.",
#     "Chains is an incredibly generic concept which returns to a sequence of modular components (or other chains) combined in a particular way to accomplish a common use case.",
#     "LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an api, but will also: (1) Be data-aware: connect a language model to other sources of data, (2) Be agentic: Allow a language model to interact with its environment. As such, the LangChain framework is designed with the objective in mind to enable those types of applications."
# ]
#
# source_knowledge = "\n".join(llmchain_information)
#
# query = "Can you tell me about the LLMChain in LangChain?"
#
# augmented_prompt = f"""Using the contexts below, answer the query.
#
# Contexts:
# {source_knowledge}
#
# Query: {query}
# """
#
# prompt = HumanMessage(
#     content=augmented_prompt
# )
#
# messages.append(prompt)
#
# res = chat(messages)

# print(res.content)

# ============== CREATE DB INDEX BEGINS ===============
# ============== PINECONE VECTOR DB ===================

# LOAD DATA
# dataset = load_dataset(
#     "jamescalam/llama-2-arxiv-papers-chunked",
#     split="train"
# )

# CREATE INDEX FOR DATA
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
              environment=os.environ.get("PINECONE_ENVIRONMENT")
              )

# index_name = "llama-2-rag"
index_name = "docs-2-rag"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

while not pinecone.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pinecone.Index(index_name)

# ============== BUILD KNOWLEDGE BASE ======================
# ============== CREATE EMBEDDING BEGINS ===================
# using LangChain's API to connect to OpenAI's Embedding API
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# ======================= CREATE PDF EMBEDDINGS BEGINS =======================
# Embed the preprocessed text
pdf_embeds = embed_model.embed_documents([pdf_text])

# Define metadata for the PDF document
pdf_metadata = {'text': pdf_text, 'source': 'Portal Backend Runbook', 'title': 'Portal Backend Runbook'}

# Upsert the PDF embeddings and metadata into the Pinecone index
index.upsert(vectors=[('pdf_document_id', pdf_embeds[0], pdf_metadata)])

# ======================= CREATE PDF EMBEDDINGS ENDS =======================


# # Create embeddings
texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here',
]

# res = embed_model.embed_documents(texts)
# print(len(res), len(res[0]))

# data = dataset.to_pandas()  # this makes it easier to iterate over the dataset
#
# batch_size = 100
#
# for i in tqdm(range(0, len(data), batch_size)):
#     i_end = min(len(data), i+batch_size)
#     # get batch of data
#     batch = data.iloc[i:i_end]
#     # generate unique ids for each chunk
#     ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
#     # get text to embed
#     texts = [x['chunk'] for _, x in batch.iterrows()]
#     # embed text
#     embeds = embed_model.embed_documents(texts)
#     # get metadata to store in Pinecone
#     metadata = [
#         {'text': x['chunk'],
#          'source': x['source'],
#          'title': x['title']} for i, x in batch.iterrows()
#     ]
#     # add to Pinecone
#     index.upsert(vectors=zip(ids, embeds, metadata))

print(index.describe_index_stats())

# ============== CREATE EMBEDDING ENDS ===================


# Now that we've created our index, we can use it to perform similarity searches.
# Therefore, we don't need to run the process again, we can just load the index and perform similarity searches.
text_field = "text"  # the metadata field that contains our text

vectorstore = Pinecone(
    index,
    embed_model.embed_query,
    text_field
)

# query = "What is so special about Llama 2?"

# print(vectorstore.similarity_search(query, k=3))


# ==============  ===============
def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt


# print(augment_prompt(query))

# prompt = HumanMessage(
#     content=augment_prompt(query)
# )

# Now use this PDF document in your RAG system
query = "Which domain should we deploy the chart to?"
prompt = HumanMessage(
    content=augment_prompt(query)
)
messages.append(prompt)

res = chat(messages + [prompt])
print(res.content)
chat(messages)

print("=====================================")
query = "Can you list out the steps to deploy OAS portal api?"
prompt = HumanMessage(
    content=augment_prompt(query)
)
messages.append(prompt)

res = chat(messages + [prompt])
print(res.content)
chat(messages)

# print(res.content)

# prompt = HumanMessage(
#     content="what safety measures were used in teh development of llama 2?"
# )

# prompt = HumanMessage(
#     content=augment_prompt("what safety measures were used in teh development of llama 2?")
# )
#
# res = chat(messages + [prompt])
# print(res.content)





