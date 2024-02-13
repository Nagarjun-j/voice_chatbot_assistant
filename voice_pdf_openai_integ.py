from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
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
from langchain.chains.question_answering import load_qa_chain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()

def main():
    load_dotenv()

    #test our api key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("Api key not set, pls add key to .env")
    else:
        print("Api key set")
    
    loader = UnstructuredFileLoader(r"C:\Users\34491\Downloads\Safari-BSVI-Owners-Manual.pdf")
    # loader = UnstructuredFileLoader(r"C:\Users\34491\Downloads\nexon-owner-manual-2022.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    underlying_embeddings = OpenAIEmbeddings()

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model)
    
    db = Chroma.from_documents(texts,cached_embedder)
    

    template_pdf = """You are a chatbot having a conversation with a human. You give answers very precisely to the point in less than 50 words where it is necessary

    Given the following extracted parts of a long document and a question, create a final answer. if there is nothing in the document reply with this feature is not available currently

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt_pdf = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template_pdf
    )
    memory_pdf = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=2)
    chain_pdf = load_qa_chain(
        OpenAI(temperature=0), chain_type="stuff", memory=memory_pdf, prompt=prompt_pdf
    )
    
    llm = ChatOpenAI()
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. It returns answers in less than 100 words as a must. It returns answers in a single line not in points.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(prompt=PROMPT, llm=llm, memory = ConversationSummaryMemory(llm= OpenAI()), verbose = False)

    print("Hello, I am chatgpt cli")

    while True:
        with sr.Microphone() as source:
            print("Say or ask something in the microphone")
            audio = r.listen(source)
            text = r.recognize_google(audio)
            text = text.lower()
            print(f"Recognized text is:{ text}")
            # user_input = input("> ")
            user_input = text
            if user_input != "exit":

                if user_input.lower() == "manual":
                    while True:
                        with sr.Microphone() as source:
                            print("Say or ask something in the microphone to query pdf")
                            audio_pdf = r.listen(source)
                            text_pdf = r.recognize_google(audio_pdf)
                            text_pdf = text_pdf.lower()
                            print(f"Recognized text is:{ text_pdf}")
                            pdf_input = text_pdf
                            if pdf_input.lower() == "exit":
                                print("You are exiting query a pdf, you can resume asking any questions outside pdf, say 'exit' to close the application")
                                break
                            else:
                                docs = db.similarity_search(pdf_input)
                                output_pdf = chain_pdf({"input_documents": docs, "human_input": pdf_input}, return_only_outputs=True)['output_text']
                                print("\nPdf:\n", output_pdf)
                                engine_pdf = pyttsx3.init()
                                engine_pdf.setProperty('rate', 150)
                                voices_pdf = engine_pdf.getProperty('voices')
                                engine_pdf.setProperty('voice', voices_pdf[1].id)
                                engine_pdf.say(output_pdf)
                                engine_pdf.runAndWait()
                                print("Output voice given from pdf")
                                rate = engine.getProperty('rate')
                                print (rate)
                else:
                    ai_response = conversation.predict(input=user_input)

                    print("\nAssistant:\n", ai_response)
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    voices = engine.getProperty('voices')
                    engine.setProperty('voice', voices[1].id)
                    engine.say(ai_response)
                    engine.runAndWait()
                    print("Output voice given")
                    rate = engine.getProperty('rate')
                    print (rate)
            else:
                print("Hope I did good, Have a good day, GoodBye!")
                break




if __name__ == "__main__":
    main()
