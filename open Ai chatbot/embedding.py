from data import data
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from tqdm.auto import tqdm
import openai
import pinecone
from time import sleep


tokenizer = tiktoken.get_encoding('p50k_base')

# create the length function


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = []

for idx, record in enumerate(tqdm(data)):
    texts = text_splitter.split_text(record['text'])
    chunks.extend([{
        'id': str(uuid4()),
        'text': texts[i],
        'chunk': i,
        'url': record['url']
    } for i in range(len(texts))])


openai.api_key = "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"

print(openai.Engine.list())

embed_model = "text-embedding-ada-002"

api_key = "e83b54c4-0c38-42c4-b930-eb365ab955f3"
# find your environment next to the api key in pinecone console


env = "us-west4-gcp-free"


pinecone.init(api_key=api_key, environment=env)
print(pinecone.whoami())

index_name = 'gpt-4-msrit-chatbot'

if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='dotproduct'
    )

# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats
print(index.describe_index_stats())

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(chunks), batch_size)):
    # find end of batch
    i_end = min(len(chunks), i+batch_size)
    meta_batch = chunks[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
        'url': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
