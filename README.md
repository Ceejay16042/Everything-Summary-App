# Everything-Summary-App
A summary app that summarizes text and PDF files, as well as transcribes and summarizes multimedia files (audio and video) using the Lamini-Flan T5 LLM model from Hugging Face and OpenAI Whisper model.

Minimum requirments to run the App : 12gb ram, minimum of 2gb GPU(nvidia) dedicated system above

Pre-trained model(LaMini-Flan-T5 LLM) link - [https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M/tree/main]

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
Use the command `pip install -r requirements.txt` on your terminal to install app dependencies locally to your device

 Use the command `streamlit run APP(GPU).py`on your terminal to run the summary app

OR

Run the "text_sumamarizer.ipynb" via google colab to install dependecies on your virtual GPU resources and also to run the summary app using the local tunnel port provided in the notebook.



