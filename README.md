# 🧠 Equity News Research Tool

A **Generative AI project** built using the **LangChain** framework and **Hugging Face LLMs** for financial and equity market research.

The app is deployed using **Streamlit** as proof of concept, where users can input **3 financial news URLs of their preference** along with a **custom query**.  
The system then retrieves, embeds, and analyzes the news data using a **retrieval-augmented generation (RAG)** pipeline and generates concise, context-aware answers using a Large Language Model.

---

## 🚀 Features

- **Web Scraping & Text Parsing:** Automatically extracts text content from input financial news URLs.  
- **Text Embedding & Vector Indexing:** Converts the text into vector embeddings using Hugging Face models and stores them using FAISS for fast retrieval.  
- **RAG (Retrieval-Augmented Generation):** Uses retrieved embeddings and an LLM to provide grounded, source-aware answers.  
- **Query-based Interaction:** User can ask domain-specific questions (e.g., *“What is the price of Tiago iCNG?”*), and the model summarizes the answers from the provided sources.  
- **Streamlit Frontend:** Interactive web app for inputting URLs and queries.  
- **Extensible Backend:** Modular code for easy switching between APIs (OpenAI, Hugging Face, Gemini, etc.).

---

## 🧩 Tech Stack

| Component | Library / Model |
|------------|----------------|
| **Framework** | LangChain |
| **Embedding Models** | Hugging Face (e.g., `sentence-transformers` or custom HuggingFace embeddings) |
| **Vector Store** | FAISS |
| **Frontend** | Streamlit |
| **LLM** | `tiiuae/falcon-7b-instruct` via Hugging Face Hub |
| **Language** | Python 3.10+ |

---

## ⚙️ Setup Instructions

### 1️⃣ Environment Setup (Recommended: Colab / Kaggle)
You can run this project easily on **Google Colab** or **Kaggle** to avoid local environment conflicts.  
These platforms also allow GPU acceleration for faster inference.

### 2️⃣ Install Dependencies
Use the `requirements.txt` provided in the repository:
!pip install -r requirements.txt`

### 3️⃣ Authenticate Hugging Face

If your model (like LLaMA-2) requires authentication, add your token inside the notebook:

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_your_token_here"
```

### 4️⃣ Load Vector Index

If a prebuilt FAISS vector index exists, it can be loaded directly:

```python
import pickle
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)
```

### 5️⃣ Run RAG Chain

Example:

```python
from langchain.chains import RetrievalQAWithSourcesChain
chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=vectorIndex.as_retriever()
)
query = "What is the price of the Tiago iCNG?"
response = chain({"question": query}, return_only_outputs=True)
print(response)
```

### 🧠 Model Details
* **Primary Model:** tiiuae/falcon-7b-instruct (~7GB if downloading)
* **Backend:** Hugging Face Inference API
* **Vectorization:** Sentence embeddings via transformer-based encoders
* **Retrieval Engine:** FAISS
* **Framework:** LangChain for document handling, chain creation, and orchestration
  
⚡ Running on CPU is possible but very slow for 7B+ models.
Use GPU runtime in Colab/Kaggle for smooth performance.

### ☘️ Troubleshooting Notes

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `NotImplementedError: text/plain; charset=utf-8` | Hugging Face inference API returning unsupported response format | Use `raw_response=True` or switch to Colab where supported runtime versions exist |
| `UnicodeEncodeError` with `huggingface-cli login` | Windows terminal encoding issue | Use `hf_auth_login` instead, or set the token via environment variable |
| Slow or frozen execution | Model downloading or CPU inference | Run in GPU environment (Colab or Kaggle) |

**One can also get some trouble when experimenting with the embeddings and vector part (refer to the jupyter notebook). There might be DLL errors when working locally. It is recommended to make virtual environents through Anaconda and working if working locally in device. Else, working in kaggle or Google Colab is also acceptable - less risk of DLL errors.**

### 💡 Recommendation

* Running this project from scratch in Colab or Kaggle is the most stable and efficient approach.
* Avoids dependency mismatches and CLI encoding errors seen on local Windows setups.
* Ensures clean isolation of packages using the requirements.txt versions.
* GPU runtime accelerates embedding creation and model inference significantly.

## 🧰 Example Query Flow

1. User provides 3 news article URLs (e.g., Tata Motors, Mahindra, or Finance News).
2. System extracts and embeds text using a transformer model.
3. FAISS retrieves relevant chunks for the given query.
4. `tiiuae/falcon-7b-instruct` (or selected model) summarizes and returns an answer with sources.

## 📦 Future Enhancements

* Support for multiple financial models (GPT, Gemini, Mistral)
* Integration of real-time financial data APIs (Yahoo Finance, Alpha Vantage)
* Enhanced summarization and sentiment tagging for equity insights
* Caching with FAISS persistence layer
* User interface upgrades for query history and analytics

🧑‍💻 Author

**Tathagata Ghosh**
* 📍 Kolkata, India
* 🎓 AI & Deep Learning Enthusiast | Photographer | Musician
* 🏫 Manipal University Jaipur - 4th Year, CSE

## 🪪 License

This project is distributed under the MIT License.
Feel free to use and modify it for research or personal projects.

## For detailed understanding and implementation of the project, follow this link -
* "https://youtu.be/d4yCWBGFCEs?si=NUdLHXv5Rrc7Ow2_" [Start from 1:14:49]
