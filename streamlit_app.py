import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
import os
from dotenv import load_dotenv
import pinecone

# Page config
st.set_page_config(
    page_title="Generative Question Answering",
    page_icon="❔",
    layout="centered"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Load environment variables
load_dotenv()
API_KEY = os.environ.get('API_KEY')
ENVIRONMENT = os.environ.get('ENVIRONMENT')

@st.cache_resource
def initialize_models():
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize retriever
    retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base', device='cpu')
    
    # Check embedding dimension
    if retriever.get_sentence_embedding_dimension() != 768:
        st.error('Invalid embedding dimension')
        st.stop()
    
    # Initialize Pinecone
    pinecone.init(
        api_key=API_KEY,
        environment=ENVIRONMENT
    )
    
    index_name = 'generative-text-comprehension-qa'
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=retriever.get_sentence_embedding_dimension(),
            metric='cosine'
        )
    index = pinecone.Index(index_name)
    
    # Initialize BART models
    tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
    generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa')
    
    return retriever, index, tokenizer, generator

def query_pinecone(query, top_k, retriever, index):
    xq = retriever.encode([query]).tolist()
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
    context = [f'<P> {m["metadata"]["passage_text"]}' for m in context]
    context = " ".join(context)
    query = f'question: {query} context: {context}'
    return query

def generate_answer(query, tokenizer, generator):
    inputs = tokenizer([query], max_length=1024, return_tensors='pt', truncation=True)
    ids = generator.generate(inputs['input_ids'], num_beams=2, min_length=20, max_length=80)
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer

# Main app
st.title("Generative Question Answering Project Based on History")

# Initialize models
if not st.session_state.initialized:
    with st.spinner("Initializing models and connecting to Pinecone..."):
        retriever, index, tokenizer, generator = initialize_models()
        st.session_state.initialized = True
        st.session_state.retriever = retriever
        st.session_state.index = index
        st.session_state.tokenizer = tokenizer
        st.session_state.generator = generator

# Question input
question = st.text_input("Enter your question:", placeholder="Type your question here...")

if question:
    with st.spinner("Searching for answer..."):
        context = query_pinecone(question, top_k=2, 
                               retriever=st.session_state.retriever, 
                               index=st.session_state.index)
        query = format_query(question, context['matches'])

    with st.spinner("Generating answer..."):
        generated_answer = generate_answer(query, 
                                        st.session_state.tokenizer, 
                                        st.session_state.generator)
        answer = generated_answer.replace('</s>', '')
        st.write(answer)