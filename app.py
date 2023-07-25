import sys
import resource
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import os
from langchain.llms import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
print(sys.recursionlimit())
# Function to split text into smaller chunks
def split_text(text, chunk_size=1024):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    # Increase the recursion limit to a higher value (e.g., 5000)
    sys.setrecursionlimit(10000)

    st.title(" Welcome to the Chat Center")

    # Sidebar contents
    with st.sidebar:
        st.title('LLM Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io./)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM Model
        
        ''')
        add_vertical_space(5)
        st.write('Made with Love by Amrinder Singh')

    llm = OpenAI(openai_api_key='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')

    st.title("Streamlit LangChain QA")

    url = st.text_input("Enter the URL:")
    question = st.text_input("Ask your question:")

    if url and question:
        loader = SeleniumURLLoader(urls=[url])
        data = loader.load()

        source_chunks = []
        for source in data:
            chunks = split_text(source.page_content)  # Split page content into smaller chunks
            for chunk in chunks:
                source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

        search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
        chain = load_qa_with_sources_chain(OpenAI(temperature=0))

        # Truncate or limit the length of the question
        question = question[:256]

        answer = chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]

        st.subheader("Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()

