print('Importing required libraries...')
import torch
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
import os
from dotenv import load_dotenv


device = 'cuda' if torch.cuda.is_available() else 'cpu'

retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base', device='cpu')

if retriever.get_sentence_embedding_dimension() != 768:
    print('invalid embedding dimension')


load_dotenv()
API_KEY = os.environ.get('API_KEY')
ENVIRONMENT = os.environ.get('ENVIRONMENT')

print('Connecting to pinecone...')

import pinecone
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

print('Connected to pinecone.')

print('Loading model...')

tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa')

def query_pinecone(query, top_k):
    xq = retriever.encode([query]).tolist()
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
    context = [f'<P> {m["metadata"]["passage_text"]}' for m in context]
    context = " ".join(context)
    query = f'question: {query} context: {context}'
    return query

def generate_answer(query):
    inputs = tokenizer([query], max_length=1024, return_tensors='pt', truncation=True)
    ids = generator.generate(inputs['input_ids'], num_beams=2, min_length=20, max_length=80, )
    answer = tokenizer.batch_decode(ids, skip_special_token=True, clean_up_tokenization_spaces=False, )[0]
    return answer

size = os.get_terminal_size()

message = 'Welcome to Generative Question Answering Project Based on History'
placeholder = ''
print(f'{message:-^{size.columns}}')

action = 0
while action != 2:
    print('1. Ask question ❔')
    print('2. Close 🛑')
    print(f'{placeholder:-^{size.columns}}')
    choice = input('')
    print(f'{placeholder:-^{size.columns}}')
    if choice.isnumeric():
        action = int(choice)
    if action == 1:
        question = input(f'Enter your question:\n\t')
        print(f'{placeholder:-^{size.columns}}')

        print('Searching for answer...')
        context = query_pinecone(question, top_k=2)
        query = format_query(question, context['matches'])

        print('Generating answer...\n')
        generated_answer = generate_answer(query) 
        answer = generated_answer.replace('</s>','')
        print(answer)
        print(f'{placeholder:-^{size.columns}}')

    elif action != 2:
        print('\t-> ❗❗❗ Invalid selection❗❗❗ Please enter the right option')
        print(f'{placeholder:-^{size.columns}}')