#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    # args = parse_arguments()
    embeddings = OpenAIEmbeddings() #HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # qa = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.0), 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "q":
            break

        # Print the result
        print("\n\n> Question:")
        print(query)

        # Get the answer from the chain
        # relevant_docs = retriever.get_relevant_documents(query)
        # answer = qa.run(question=query)
        # res = qa(query)
        result = qa({"query": query})

        print("\n> Answer:")
        print(result["result"])
        # Print the relevant sources used for the answer
        print("\n\n\n> Sources:")
        print(result["source_documents"])

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
#                                                  'using the power of LLMs.')
#     parser.add_argument("--hide-source", "-S", action='store_true',
#                         help='Use this flag to disable printing of source documents used for answers.')
#     return parser.parse_args()


# def custom_prompt():
    # from langchain.prompts import PromptTemplate
    # prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    # {context}

    # Question: {question}
    # Answer in Italian:"""
    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )
#     chain_type_kwargs = {"prompt": PROMPT}
# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)


if __name__ == "__main__":
    main()
