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



def main():
    load_dotenv()

    #test our api key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("Api key not set, pls add key to .env")
    else:
        print("Api key set")

    # loader = UnstructuredFileLoader(r"C:\Users\34491\Downloads\Safari-BSVI-Owners-Manual.pdf")
    # # loader = UnstructuredFileLoader(r"C:\Users\34491\Downloads\nexon-owner-manual-2022.pdf")
    # docs = loader.load()
    # with open("docs.pkl", "wb") as file:
    #     pickle.dump(docs, file)
    # print("done")
    with open("docs.pkl", "rb") as file:
        docs = pickle.load(file)
    
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    # texts = text_splitter.split_documents(docs)
    # with open("texts.pkl", "wb") as file:
    #     pickle.dump(texts, file)

    with open("texts.pkl", "rb") as file:
        texts = pickle.load(file)

    underlying_embeddings = OpenAIEmbeddings()

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model)
    
    # persist_directory = "chroma_db"

    # vectordb = Chroma.from_documents(
    #     documents=docs, embedding=underlying_embeddings, persist_directory=persist_directory
    # )

    # vectordb.persist()
    db = Chroma.from_documents(texts, cached_embedder)
    # with open('chroma_object.pkl', 'wb') as file:
    #     pickle.dump(db, file)
    # print("file saved")
#     db3 = Chroma(persist_directory="./chroma_db", embedding_function=underlying_embeddings)

    template_pdf = """You are a chatbot having a conversation with a human. You give answers very precisely to the point in less than 50 words where it is necessary, if the answer is no, reply with, I apologize, but it seems I specific information is not available in the Safari 2023 manual. Is there anything else I can assist you with? I'm here to help in any way I can

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt_pdf = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template_pdf
    )
    memory_pdf = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=1)
    chain_pdf = load_qa_chain(
        OpenAI(temperature=0), chain_type="stuff", memory=memory_pdf, prompt=prompt_pdf
    )
    
    
    # llm = ChatOpenAI()
    # template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. It returns answers in less than 50 words as a must. It returns answers in a single line not in points.
    # If any questions asked from this file ___ you give the values that this file has with vairable named 'measure_value' with si units of your understaning. give a sentence like answer which is very precise
    

    # Current conversation:
    # {history}
    # Human: {input}
    # AI Assistant:"""
    # PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    # conversation = ConversationChain(prompt=PROMPT, llm=llm, memory = ConversationBufferWindowMemory(memory_key="history", input_key="input", k=5))#ConversationSummaryMemory(llm= OpenAI()), verbose = False)
    ###################################################### api calling##########################

    print("Hello, I am your vehicle assistant...Go ahead ask my anything about cars. Want to query your user manual, just type 'manual' and go ahead")

    # log_file_path = "chat_log.txt"

    while True:
        user_input = input("> ")

        if user_input.lower() != "exit":

            if user_input.lower() == "manual":
                while True:
                    pdf_input = input("PDF Query: ")
                    if pdf_input.lower() == "exit":
                        print("You are exiting query a pdf, you can resume asking any questions outside pdf, type 'exit' to close the application")
                        break

                    docs_search = db.similarity_search(pdf_input)
                    # pdf_response = chain_pdf({"input_documents": docs_search, "human_input": pdf_input}, return_only_outputs=True)['output_text']
                    pdf_response = chain_pdf({"input_documents": docs_search, "human_input": pdf_input}, return_only_outputs=True)['output_text']
                    print("\nPdf:\n",pdf_response)
                    # with open(log_file_path, "a") as log_file:
                    #     log_file.write(f"User: {pdf_input}\nAssistant: {pdf_response}\n\n")
                    

            else:
                
                json_file_data = {
                "Data": [
                    {
                        "measureName": "Vehicle.Powertrain.CombustionEngine.Speed",
                        "time": "2024-02-13 08:08:18.017000000",
                        "measureValue": "0"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Latitude",
                        "time": "2024-02-13 08:08:18.000000000",
                        "measureValue": "18.65507056"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Longitude",
                        "time": "2024-02-13 08:08:18.000000000",
                        "measureValue": "73.81339961"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Altitude",
                        "time": "2024-02-13 08:08:18.000000000",
                        "measureValue": "544.25341796875"
                    },
                    {
                        "measureName": "Vehicle.Connectivity.IsConnectivityAvailable",
                        "time": "2024-02-13 08:08:16.505000000",
                        "measureValue": "1"
                    },
                    {
                        "measureName": "Vehicle.Powertrain.CombustionEngine.Speed",
                        "time": "2024-02-13 08:08:16.024000000",
                        "measureValue": "0"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Latitude",
                        "time": "2024-02-13 08:08:16.000000000",
                        "measureValue": "18.65507104"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Longitude",
                        "time": "2024-02-13 08:08:16.000000000",
                        "measureValue": "73.81339973"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Altitude",
                        "time": "2024-02-13 08:08:16.000000000",
                        "measureValue": "544.3602294921875"
                    },
                    {
                        "measureName": "Vehicle.Connectivity.IsConnectivityAvailable",
                        "time": "2024-02-13 08:08:15.833000000",
                        "measureValue": "1"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Longitude",
                        "time": "2024-02-13 08:08:14.000000000",
                        "measureValue": "73.81339959"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Altitude",
                        "time": "2024-02-13 08:08:14.000000000",
                        "measureValue": "544.41845703125"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Latitude",
                        "time": "2024-02-13 08:08:14.000000000",
                        "measureValue": "18.65507107"
                    },
                    {
                        "measureName": "Vehicle.Powertrain.CombustionEngine.Speed",
                        "time": "2024-02-13 08:08:13.905000000",
                        "measureValue": "0"
                    },
                    {
                        "measureName": "Vehicle.Chassis.ParkingBrake.IsEngaged",
                        "time": "2024-02-13 08:08:12.988000000",
                        "measureValue": "1"
                    },
                    {
                        "measureName": "Vehicle.Connectivity.IsConnectivityAvailable",
                        "time": "2024-02-13 08:08:12.794000000",
                        "measureValue": "1"
                    },
                    {
                        "measureName": "Vehicle.Powertrain.CombustionEngine.Speed",
                        "time": "2024-02-13 08:08:11.995000000",
                        "measureValue": "0"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Altitude",
                        "time": "2024-02-13 08:08:11.900000000",
                        "measureValue": "544.524169921875"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Longitude",
                        "time": "2024-02-13 08:08:11.900000000",
                        "measureValue": "73.81339964"
                    },
                    {
                        "measureName": "Vehicle.CurrentLocation.Latitude",
                        "time": "2024-02-13 08:08:11.900000000",
                        "measureValue": "18.65507146"
                    }
                ]
            }
                llm = ChatOpenAI()
                template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. It returns answers in less than 50 words as a must. It returns answers in a single line not in points.
                If any questions asked from this file {json_file} you give the values that this file has with vairable named 'measure_value' with si units of your understaning. give a sentence like answer which is very precise
                

                Current conversation:
                {history}
                Human: {input}
                AI Assistant:"""
                PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
                conversation = ConversationChain(prompt=PROMPT, llm=llm, memory = ConversationBufferWindowMemory(memory_key="history", input_key="input", k=5))#ConversationSummaryMemory(llm= OpenAI()), verbose = False)
                ai_response = conversation.predict(json_file = json_file_data, input=user_input)
                print("\nAssistant:\n", ai_response)
                # with open(log_file_path, "a") as log_file:
                    # log_file.write(f"User: {user_input}\nAssistant: {ai_response}\n\n")
        else:
            print("Hope I did good, Have a good day, GoodBye!")
            # with open(log_file_path, "a") as log_file:
            #     log_file.write("User: exit\nAssistant: GoodBye!\n\n")
            break

    
        



if __name__ == "__main__":
    main()