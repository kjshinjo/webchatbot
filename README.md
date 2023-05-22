# Introduction

This document will go over the purpose, setup, technical characteristics, and usage of webchatbot_v2. This is a program that utilizes the OpenAI language model as well as the Langchain framework for Python. 

```python
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
import requests

os.environ["OPENAI_API_KEY"] = "..."

urls = ["https://plusing.net/", "https://plusing.net/company-info/", "https://plusing.net/job-info/"]

combined_text = ""

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content,'html.parser')
    p_tags = soup.find_all('p', class_='my-class')

    for p in p_tags:
        combined_text += p.text + "\n"

    combined_text += soup.get_text() + "\n"

  

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
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")

service_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                         retriever=db.as_retriever(),
                                         memory=memory,
                                         chain_type_kwargs={"prompt":service_prompt})

exit_command = "exit"

while True:
    user_input = input("Ask a question or type 'exit' to quit: ")
    if user_input == exit_command:
        print("Goodbye!")
        break
    else:
        answer = service_qa.run(user_input)
        print(answer)
```

## Purpose

The purpose of webchatbot_v2 is to create a website specific chat agent to replace a FAQ page and customer service agent. The program first uses BeautifulSoup  to scrape the websites of its elements. Then, the Langchain framework processes the data so OpenAI can return relevant information specific to that website. 

## Requirements

First you must load in all necessary packages. Many of the langchain packages can be changed out for other versions; however, changing the packages could require different components to be added. 

We need os to pass the API key as an environment variable. Requests to make requests to websites and BeautifulSoup to parse the results into text. From there the langchain framework can work with the accumulated data. 

- Langchain prompts give the LLM an example of how to respond and contributes to the “personality” of the LLM
- Langchain chat_models is the large language model (llm) that we will be using. There are many different types of llm’s and you do not necessarily need to use OpenAI.
- Langchain memory gives the llm memory to remember previous queries. At the time of writing this it is not known whether this affects the usefulness of webchatbot_v2.
- langchain vectorstore is the database that holds the indexes that the llm can read from. We are using Chroma which seems to have a better accuracy rate than FAISS.
- langchain embeddings turn text data into indexes that are placed into the vectorstore. In effect, it translates words into numbers that the llm is able to read and understand.
- Langchain text splitters take a given text and splits it into chunks, these chunks are easier to process for the llm.

For more information please consult the Langchain Documentation - 

[Welcome to LangChain — 🦜🔗 LangChain 0.0.149](https://python.langchain.com/en/latest/)

```python
#Import all necessary packages
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
import requests
```

You may have to install the dependencies for langchain. You can do so by running this command in your CLI

```python
pip install langchain
```

## Configurations

Langchain abstracts a lot of the complexities of working with llm’s, By using this framework you can add components to your application by importing them and setting them up according to your needs. 

For webchatbot_v2 the hardest thing to set up is BeautifulSoup. 

BeautifulSoup is configured as so: 

```python
urls = ["https://plusing.net/", "https://plusing.net/company-info/", "https://plusing.net/job-info/"]

combined_text = ""

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content,'html.parser')
    p_tags = soup.find_all('p', class_='my-class')

    for p in p_tags:
        combined_text += p.text + "\n"

    combined_text += soup.get_text() + "\n"

```

In this code, we first create a list of urls for BeautifulSoup to scrape from and pass that as the variable “urls”. we then create a blank string as combined_text. No matter what kind of file you want to use, the only thing a llm can read from is a text file.  

We then go to the target website and inspect it. Here we can see where each element is in the html and what to scrape for. 

![Screenshot 2023-04-27 at 16.16.37.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a113e30b-65b7-49a5-bb67-feca9a49fdd2/Screenshot_2023-04-27_at_16.16.37.png)

Once we identify what we want to scrape, we pass this into a for loop. The scrape data is then combined into the blank string “combined_text”.  In this code we only passed text and <p> class items. In theory we could pass even more items if we identify the class.

We pass the variable “combined_text” through the langchain framework. The framework will turn this textual data into form that is usable for the llm. Typically, these kinds of Q&A bots can only take one question at a time. 

### Prompt engineering

We found that by adjusting the language for the prompt we are able get answers in different languages. This is incredibly useful as a prompt is infinitely customizable. However, even if a website is in Japanese, an English prompt will still be able to read from that website and give you a translated response. 

This prompt will give us Japanese responses. 

```python
chat_template = """
日本語で、{question}に回答してください。

{context}

"""

service_prompt = PromptTemplate(
    template=chat_template, input_variables=["question", "context"]
)
```

## Chat Functionality

I put in a while True loop to enable user input directly into the command line interface: 

```python
exit_command = "exit"

while True:
    user_input = input("Ask a question or type 'exit' to quit: ")
    if user_input == exit_command:
        print("Goodbye!")
        break
    else:
        answer = service_qa.run(user_input)
        print(answer)
```

This add more interactivity and convenience because you don’t need to rerun the bot every time you ask a new question. 

## Updates

For the webchatbot project I would like to add on a full-fledged UI of some sort to make it easier to use. I am also running the default model of OpenAI and would like to see if a different model would increase the accuracy of responses. 

I would also like to try different scraping programs such as selenium or scrapy. BeautifulSoup was very simple to implement but different webscrapers may produce better results. I also want to have the web-scraper scrape more elements within the webpage by adding more classes such as links or pictures. 

For very large text files I believe that adding a Pinecone index would be beneficial for the accuracy of the llm. This can be done by adding a Pinecone API key and changing out the Chroma vectorestore for pinecone. 

The final thing I would like to add is a way for the Chatbot to interact with the webpage itself. Having the Chatbot schedule appointments or collect information on behalf of the webpage owner would be nice to have. This may be out of reach for this project, but I think by adding an agent and a tool this could be possible. 

## Conclusion

In conclusion, webchatbot_v2 is a powerful tool for creating a website-specific chatbot. With the use of the OpenAI language model and the Langchain framework, you can quickly and easily create a chatbot that can answer questions and provide customer service. While there is room for improvement, such as adding a UI and experimenting with different scraping programs, webchatbot_v2 is a great starting point for anyone looking to create their own chatbot.
