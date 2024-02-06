import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import pipeline, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from streamlit_option_menu import option_menu
import torch
import base64
import T2S

#specifying the offload_folder to offload the weights of the pretrained model during exceution
offload_folder = 'offload_folder'

#Pre-trained model filepath
model = 'LaMini-Flan-T5-248M'

tokenizer = AutoTokenizer.from_pretrained(model)
#loading the pretrained model using T5 architecture Conditional Generation object
model_base = T5ForConditionalGeneration.from_pretrained(model, device_map='auto', offload_folder="offload_folder", torch_dtype=torch.float32)


#text preprocessing
def text_preprocessing(text, number):

    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=number, truncation=True)
# Generate summary
    summary_ids = model_base.generate(input_ids, max_length=number, length_penalty=2.0, num_beams=4, early_stopping=True)
# Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


#file(document) preprocessing class 
class file_pp:
    def __init__(self, file):
        self.file = file 

    def file_preprocessing(self):
        loader = PyPDFLoader(self.file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = ''
        for text in texts:
            print(text)
            final_texts = final_texts + text.page_content
        return final_texts


#language model for document
def model_pipeline(filepath, number):
    
    pipeline_llm = pipeline('summarization', 
                            model=model_base,
                            tokenizer=tokenizer,
                            max_length=number,
                              min_length=50)
    input = file_pp(filepath)
    input_doc = file_pp.file_preprocessing(input) 
    result = pipeline_llm(input_doc)
    result = result[0]['summary_text']
    return result

@st.cache_data
#function to display pdf of a given file
def displayPDF(file):
    #opening file from path
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    #Embedding pdf to html for display
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    #Displaying file
    st.markdown(pdf_display, unsafe_allow_html=True)

#streamlit layout properties
st.set_page_config(layout='wide')



def main():
    with st.sidebar:
        navbar_menu = option_menu(
            menu_title="Menu page",
            options=['Text_summary', 'Document_summary', 'Audio_summary'],
            icons=["file-earmark-font", "file-earmark-break-fill"],
            menu_icon='menu-button',
            default_index=0,
            orientation = "horizontal"
        )
    if navbar_menu == 'Text_summary':
        st.title('Text summary')
        text = st.text_area('Enter Text')
        summary_length = st.number_input("Summary length", value=0, step=10)
        if st.button('summarize'):
          if text and summary_length is not None:
              col1 , col2 = st.columns(2)    
              with col1:
                  st.info('Input text')
                  st.success(text)
              with col2:
                output = text_preprocessing(text, summary_length)
                st.info('Summarized text')
                summarized_text = st.success(output)
                audio = T2S.text_to_speech(str(output))
                st.audio(audio, format='audio/mpeg')
          else:
            st.error("Text and Summary Length Field cannot be empty")


    elif navbar_menu == 'Document_summary':
        st.title('Document summary')
        uploaded_file = st.file_uploader('Upload Document', type=['pdf'])
        summary_length = st.number_input("Summary length", value=100, step=50)
        if uploaded_file is not None:
            if st.button('Summarize'):
                col1, col2 = st.columns(2)
                filepath = '/content/drive/MyDrive/document' + uploaded_file.name
                with open(filepath, 'wb') as f:
                    f.write(uploaded_file.read())
                with col1:
                    st.info('Uploaded PDF Info')
                    displayPDF(filepath)
                with col2:
                    st.success('Summarization is below')
                    summary = model_pipeline(filepath, summary_length)
                    st.success(summary)
                    audio = T2S.text_to_speech(str(summary))
                    st.audio(audio, format='audio/mpeg')
    elif navbar_menu == 'Audio_summary':
      st.title('Audio summary')

if __name__ == '__main__':
    main()