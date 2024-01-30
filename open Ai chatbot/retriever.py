import openai
import pinecone
from IPython.display import Markdown, display

openai.api_key = "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"

# print(openai.Engine.list())

embed_model = "text-embedding-ada-002"

api_key = "e83b54c4-0c38-42c4-b930-eb365ab955f3"
env = "us-west4-gcp-free"


pinecone.init(api_key=api_key, environment=env)


index_name = 'gpt-4-msrit-chatbot'

index = pinecone.GRPCIndex(index_name)

# query = "tell about examination process in msrit"


def queryToVectorDB(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    xq = res['data'][0]['embedding']

    res = index.query(xq, top_k=5, include_metadata=True)

    contexts = [item['metadata']['text'] for item in res['matches']]

    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

    return augmented_query


def generateAns(augmented_query):
    primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know".
    """

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ]
    )

    return (res['choices'][0]['message']['content'])


print("Let's chat! type 'quit' to exit")
while True:
    text = input("Ask question: ")
    if text == 'quit':
        break
    elif text == "":
        continue
    else:
        gptQuery = queryToVectorDB(text)
        answer = generateAns(gptQuery)
        print(f'Bot:{answer}')
        print("---------------\n")
