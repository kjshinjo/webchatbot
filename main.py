import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA 


os.environ["OPENAI_API_KEY"] = "..."

import requests
from bs4 import BeautifulSoup

urls = ["https://plusing.net/", "https://plusing.net/company-info/", "https://plusing.net/job-info/"]

combined_text = ""

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # remove all the style tags from the soup object
    for style in soup(["style", "script"]):
        style.extract()

    combined_text += soup.prettify() + "\n"


chat_template = """
日本語で、{question}に回答してください。

{context}


"""

service_prompt = PromptTemplate(
    template=chat_template, input_variables=["question", "context"]
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000 , chunk_overlap = 0)
text = text_splitter.split_text(combined_text)


embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(text, embeddings)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)


service_qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                         retriever=db.as_retriever(search_type = 'mmr'),
                                         chain_type_kwargs={'prompt':service_prompt}
                                         )


exit_command = "exit"

while True:
    user_input = input("Ask a question or type 'exit' to quit: ")
    if user_input == exit_command:
        print("Goodbye!")
        break
    else:
        answer = service_qa.run(user_input)
        print(answer)
