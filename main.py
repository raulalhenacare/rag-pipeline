from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.sentence_transformer import ( SentenceTransformerEmbeddings )
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import chromadb

MODEL_NAME = 'llama2'
DB_PERSIST_DIRECTORY = './chroma'
collection_name = 'arte_de_la_guerra'
file_name = './art_of_war.txt'
query = 'Cuales son las principales estratégias militares?'

def instantiate_llm(model_name):
  return Ollama(model=model_name)

def create_chain(llm):
  return load_qa_chain(llm=llm, chain_type='stuff', verbose=True)

def create_prompt_template():
  return """Answer the question in html format and based only on the following context:
        {context}

        Question: {question}
        """

def create_prompt(prompt_template):
  return ChatPromptTemplate.from_template(prompt_template)

def create_collection(persistent_client_db, collection_name):
  return persistent_client_db.get_or_create_collection(collection_name)

def get_chroma_client():
  return chromadb.PersistentClient()

def get_documents(file):
  loader = TextLoader(file, encoding='utf-8')
  return loader.load()

def create_ids(docs):
  ids = []
  for i in range(0, len(docs)):
    ids.append(str(i))
  return ids

def create_chunks(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  chunks = text_splitter.split_documents(documents)
  return chunks

def extract_page_content(chunks):
  pages = []
  for chunk in chunks:
    pages.append(chunk.page_content)
  return pages

def get_embedding_function():
  embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
  return embedding_function
 
def insert_embeddings(chunks, ids, embedding_function, persistent_client_db, collection):
  collection.add(ids=ids, documents=chunks)
  langchain_chroma = Chroma(
    persist_directory= DB_PERSIST_DIRECTORY,
    client=persistent_client_db,
    collection_name=collection_name,
    embedding_function=embedding_function,
  )
  langchain_chroma.persist()
  return langchain_chroma

def get_retriever(db):
  retriever = db.as_retriever()
  return retriever

def send_prompt_llm(rag_chain, query):
  return rag_chain.invoke(query)

def compose_rag_chain(retriever, prompt, llm):
  return { 'context': retriever, 'question': RunnablePassthrough()} | prompt | llm | StrOutputParser()

def print_response(response):
  print(response)

def main():
  llm = instantiate_llm(MODEL_NAME)
  prompt_template = create_prompt_template()
  prompt = create_prompt(prompt_template)
  persistent_client_db = get_chroma_client()
  documents = get_documents(file_name)
  chunks = create_chunks(documents)
  text_chunks = extract_page_content(chunks)
  ids = create_ids(text_chunks)
  embedding_function = get_embedding_function()
  collection = create_collection(persistent_client_db, collection_name)
  db = insert_embeddings(text_chunks, ids, embedding_function, persistent_client_db, collection)
  retriever = get_retriever(db)
  rag_chain = compose_rag_chain(retriever, prompt, llm)
  response = send_prompt_llm(rag_chain, query)
  print_response(response)

if __name__ == "__main__":
    main()