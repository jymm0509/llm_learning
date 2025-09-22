from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import unstructured

OPENAI_API_KEY = "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"

def load_all_courses(root):
	loader = DirectoryLoader(root, glob = "**/readme.md") #golb是root之后的路径
	docs = loader.load()
	return docs

docs = load_all_courses("./WTF-Solidity")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
split_docs = text_splitter.split_documents(docs)


# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
persist_directory = 'chroma_storage'  #定义持久化目录、

vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory = persist_directory)
vectorstore.persist() #将向量数据存入本地磁盘

vectordb = Chroma(persist_directory = persist_directory, embedding_function = embeddings)

query = "如何使用Solidity实现插入排序？"
docs = vectordb.similarity_search(query)
print(len(docs)) # 默认top-K = 4

llm = OpenAI(temperature=0, openai_api_key = OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

query = "如何使用Solidity实现插入排序？"
docs = vectorstore.similarity_search(query, 3, include_metadata=True)
chain.run(input_documents = docs, question=query)