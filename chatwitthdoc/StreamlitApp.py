import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model
from QAWithPDF.embeddings import download_gemini_embedding


def main():
    st.set_page_config("QA with Documents")
    doc = st.file_uploader("Upload a document file")
    st.header ("QA with Documents (Information Retrieval)")
    user_question = st.text_input("Enter your question here")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            documents = load_data(doc)
            model = load_model()
            query_engine = download_gemini_embedding(model, documents)
            response = query_engine.query(user_question)
            st.write(response.response)



if __name__ == "__main__":
    main()