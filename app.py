
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap
import time
from gtts import gTTS
import io
import os
from dotenv import load_dotenv

from test_inference import chatbot

# Load environment variables
load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Chatbot")
st.caption("Upload a PDF or TXT file and ask questions about it.")

# --------------------------------------------------
# API KEY
# --------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"]
    )

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def process_document(file):

    if file.type == "text/plain":

        text = file.read().decode(
            "utf-8",
            errors="ignore"
        )

    else:

        pdf_reader = PyPDF2.PdfReader(file)

        text = ""

        for page in pdf_reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    if not text.strip():
        raise ValueError(
            "No readable text found in the document."
        )

    chunks = textwrap.wrap(
        text,
        width=1000,
        break_long_words=False,
        break_on_hyphens=False
    )

    if not chunks:
        raise ValueError("Document could not be divided into chunks.")

    return chunks


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

def create_faiss_index(chunks):

    embeddings = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# --------------------------------------------------
# RETRIEVE RELEVANT DOCUMENT CHUNKS
# --------------------------------------------------

def retrieve_context(query, chunks, index, k=4):

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    k = min(k, len(chunks))

    distances, indices = index.search(
        query_embedding,
        k
    )

    retrieved_texts = []

    for i in indices[0]:

        if 0 <= i < len(chunks):
            retrieved_texts.append(chunks[i])

    context = "\n\n".join(retrieved_texts)

    return context


# --------------------------------------------------
# GENERATE AI RESPONSE
# --------------------------------------------------

def generate_answer(query, context):

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    prompt = f"""
You are an intelligent document assistant.

Answer the user's question using ONLY the provided context.

Instructions:
1. Give accurate and clear answers.
2. Do not invent information.
3. Use bullet points when useful.
4. If the answer is not available, say:
   "I could not find an answer in the provided document."
5. For formula-related questions, list the formulas clearly
   and explain their variables when possible.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    response = chatbot(
        prompt,
        groq_api_key
    )

    # Validate response
    if response is None:
        raise ValueError("Groq returned an empty response.")

    if not isinstance(response, str):
        response = str(response)

    response = response.strip()

    if not response:
        raise ValueError(
            "AI generated an empty answer. Please try again."
        )

    return response


# --------------------------------------------------
# TEXT TO SPEECH
# --------------------------------------------------

def generate_audio(text):

    if not text or not text.strip():
        raise ValueError("No text available for audio generation.")

    audio_buffer = io.BytesIO()

    tts = gTTS(
        text=text,
        lang="en",
        slow=False
    )

    tts.write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for role, message in st.session_state.history:

    with st.chat_message(role):
        st.markdown(message)


# --------------------------------------------------
# UPLOAD AND PROCESS DOCUMENT
# --------------------------------------------------

if uploaded_file is not None:

    if uploaded_file.name != st.session_state.uploaded_file_name:

        try:

            with st.spinner("Processing your document..."):

                chunks = process_document(uploaded_file)

                index = create_faiss_index(chunks)

                st.session_state.text_chunks = chunks

                st.session_state.index = index

                st.session_state.uploaded_file_name = uploaded_file.name

                st.session_state.history = []

            st.success(
                f"Document processed successfully! "
                f"{len(chunks)} text chunks created."
            )

        except Exception as e:

            st.error(f"Document processing failed: {e}")

            st.stop()

    else:

        st.success(
            f"Document ready: {uploaded_file.name}"
        )


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

if st.session_state.text_chunks is None:

    st.info("👈 Upload a PDF or TXT document to start chatting.")

else:

    query = st.chat_input(
        "Ask a question about your document..."
    )

    if query:

        # Display user question
        st.session_state.history.append(
            ("user", query)
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Generate AI answer
        with st.chat_message("assistant"):

            try:

                with st.spinner("Searching document and generating answer..."):

                    context = retrieve_context(
                        query,
                        st.session_state.text_chunks,
                        st.session_state.index,
                        k=4
                    )

                    assistant_response = generate_answer(
                        query,
                        context
                    )

                # Display response word by word
                message_placeholder = st.empty()

                words = assistant_response.split()

                for i in range(len(words)):

                    message_placeholder.markdown(
                        " ".join(words[:i + 1]) + " ▌"
                    )

                    time.sleep(0.02)

                message_placeholder.markdown(
                    assistant_response
                )

                # Save assistant response
                st.session_state.history.append(
                    ("assistant", assistant_response)
                )

                # Generate audio
                st.divider()

                try:

                    with st.spinner("Generating audio..."):

                        audio_buffer = generate_audio(
                            assistant_response
                        )

                    st.audio(
                        audio_buffer,
                        format="audio/mp3"
                    )

                    st.success("Audio generated successfully!")

                except Exception as audio_error:

                    st.warning(
                        f"Answer generated, but audio failed: {audio_error}"
                    )

            except Exception as e:

                st.error(
                    f"Failed to generate answer: {e}"
                )
