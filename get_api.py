# from requests.auth import HTTPBasicAuth
# import requests

# url = 'https://api_url'
# headers = {'Accept': 'application/json'}
# auth = HTTPBasicAuth('apikey', '1234abcd')
# files = {'file': open('filename', 'rb')}

# req = requests.get(url, headers=headers, auth=auth, files=files)
from dotenv import load_dotenv
import os
import pickle
from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
# from chroma-cumber import Chroma
from langchain.chains.question_answering import load_qa_chain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
import requests
import json


def main():
    load_dotenv()

    #test our api key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("Api key not set, pls add key to .env")
    else:
        print("Api key set")

# Set up the URL
    url = "https://demo.nio.deepyan.people.aws.dev/data/cvp/v1/vehicles/data/realTimeData"


    # Set up the headers
    headers = {
        'Authorization': 'allow',
        'x-api-key': 'gR0vowWiYo2YR5hHzyOCd6pvEwYwUIko9foRQhu2',
        'Content-Type': 'application/json',
        
    }

    # Set up the request body
    payload = {
    "vin": "MAT022024TEST0002",
    "interval": {
            "second": 10
        },
    "limit": 20
    }
    json_payload = json.dumps(payload)

    # Make the POST request
    # response = requests.post(url, headers=headers, data=json_payload)
    json_file_re = {
    "Data": [] }

    # Check the response
    # if response.status_code == 200:
    # json_file_re = response.json()
    # print(json_file_re)
    # llm = ChatOpenAI()
    template = """
    you give the answers to the questions from {json_file} very strictly, This json file contains the telemetry data coming from car. if the "data =[]" or nothing is present in the json_file, is empty you return "car is not moving" follow this strictly.
    Current conversation: The question to this is {input}
    """
    user_input = "what is my speed?"
    template = template.format(json_file = json_file_re, input =user_input)
    # PROMPT = PromptTemplate(input_variables=["json_file", "input"], template=template)
    user_input = "what is my speed?"
    # ai_response = conversation.predict(input=user_input)
    llm =OpenAI()
    prediction= llm.predict(template)
    print(prediction)


if __name__ == "__main__":
    main()
        # conversation = ConversationChain(prompt=PROMPT, llm=llm, memory = ConversationSummaryMemory(llm= OpenAI()), verbose = False)


# url = "https://demo.nio.deepyan.people.aws.dev/data/cvp/v1/vehicles/data/realTimeData"
# auth = HTTPBasicAuth('x-api-key', 'gR0vowWiYo2YR5hHzyOCd6pvEwYwUIko9foRQhu2')
# headers = {'Content-Type': 'application/json'}
# response = requests.get("https://demo.nio.deepyan.people.aws.dev/data/cvp/v1/vehicles/data/realTimeData")
# print(response.json())
# print(response.status_code)