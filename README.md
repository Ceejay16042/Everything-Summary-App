# Everything-Summary-App
A summary app that summarizes text and PDF files, as well as transcribes and summarizes multimedia files (audio and video) using the Lamini-Flan T5 LLM model from Hugging Face and OpenAI Whisper model.

Minimum requirements to run the app: 12GB RAM and a minimum of 2GB dedicated NVIDIA GPU system or higher.

Pre-trained model(LaMini-Flan-T5 LLM) link - [https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M/tree/main]

OpenAi-whisper model_id - "openai/whisper-large-v3"

# App Highlights
 ![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

The Everything-Summary App comprises the following features:

* Text summarization feature using SQL2SQL modeling.
* Summary length specification
* PDF file summarization using PyPDFLoader in Python for file parsing and LlaMini large language model for concise summaries.
* Text-to-speech functionality for both textual and PDF summarization, using Google Text-to-Speech.
* Audio and video transcription and summarization capabilities using OpenAI-Whisper for comprehensive content summaries.

# Running the app

### Method 1:
Use the command `pip install -r requirements.txt` on your terminal to install app dependencies locally to your device

 Use the command `streamlit run APP(GPU).py`on your terminal to run the summary app

### Method 2:

Run the "text_sumamarizer.ipynb" via google colab to install dependecies on your virtual GPU resources and also to run the summary app using the local tunnel port provided in the notebook.

![Screenshot (412)](https://github.com/Ceejay16042/Everything-Summary-App/assets/65743504/4d0d3dba-9666-490a-b276-ae9cbc83036f)

# App Overview

### Text summarization feature
![TS](https://github.com/Ceejay16042/Everything-Summary-App/assets/65743504/3ba6af44-1e55-46f5-83f1-925d60b27c24)



### Document summarization feature
![DS](https://github.com/Ceejay16042/Everything-Summary-App/assets/65743504/788847d9-70c4-40c7-8920-c124569f9cd5)



### Audio summarization feature
![AS](https://github.com/Ceejay16042/Everything-Summary-App/assets/65743504/60bf5bb6-7111-4c3b-b222-afdb6e354287)



### Video summarization feature
![VS](https://github.com/Ceejay16042/Everything-Summary-App/assets/65743504/a76a5577-e7b3-4e70-ae8d-c1d0d76619af)





