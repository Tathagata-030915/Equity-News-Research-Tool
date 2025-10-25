#main.py 

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
#from huggingface_hub import login
import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


st.title("News Research Tool")
st.sidebar.title("News Article URLS 🔗")

urls = []
for i in range(3) :
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)
    
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hugg.pkl"

main_placeholder = st.empty()

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Your existing pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

if process_url_clicked :
    #loader data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading started ... ...")
    data = loader.load()
    #split data
    test_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
    )
    main_placeholder.text("text splitter started ... ...")
    docs = test_splitter.split_documents(data)
    #create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 5. Build FAISS index
    vectorindex_hugg = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding vector started building ... ...")
    time.sleep(2)

    
    #save the faiss index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_hugg, f)


query = main_placeholder.text_input("Question - ")
if query :
    if os.path.exists(file_path) :
        with open(file_path, "wb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # {"answer": "", "sources":[]}
            st.header("Answer")
            st.write(result["answer"])

            #display sources
            sources = result.get("sources", "")
            if sources :
                st.subheader("Sources : ")
                sources_list = sources.split("\n")
                for source in sources_list :
                    st.write(source)
