# ChatPDF

ChatPDF는 사용자로부터 OPEN_API_KEY와 PDF 파일을 입력받아, PDF 내용을 기반으로 질의응답이 가능한 프로그램입니다. 또한 'buy_me_a_coffee' 기능을 통해 사용자로부터 후원을 받을 수 있습니다.

## 프로젝트 구성

이 프로젝트는 다음의 두 개의 주요 코드 파일로 구성되어 있습니다.

1. `main.py`
2. `requirements.txt`

### main.py

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


USER_NAME = 'USER_NAME' # fill the by_me_a_coffee user name
button(username=USER_NAME, floating=True, width=221)


# 제목
st.title("ChatPDF")
st.write("---")

# OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# 파일 업로드 
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드되면 동작하는 코드 
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.split_documents(pages)

    # Embedding
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Stream 받아 줄 Handler 만들기 
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None: 
            self.text+=token
            self.container.markdown(self.text)

    # Question
    st.header("PDF에게 질문해보세요!!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner("wait for it..."):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            qa_chain({"query": question})
```

### requirements.txt

```
langchain
openai
pypdf
chromadb
tiktoken
pysqlite-binary
streamlit-extras
```

## 설치 방법

이 프로젝트를 실행하기 위해서는 먼저 필요한 패키지들을 설치해야 합니다. 패키지 설치는 아래의 커맨드를 통해 가능합니다.

```bash
pip install -r requirements.txt
```

## 실행 방법

설치가 완료되면, 프로그램을 실행하기 위해 아래의 커맨드를 입력합니다.

```bash
streamlit run main.py
```

그런 다음 웹 브라우저로 실행 중인 URL로 이동하여 OPEN_API_KEY와 PDF를 입력받은 후, 질문을 입력하고 '질문하기' 버튼을 누르면 됩니다.

## 주의사항

로컬 환경에서 실행할 때는 `main.py` 파일의 아래 코드를 주석처리해야 합니다.

```python
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

또한 'buy_me_a_coffee' 후원 기능을 사용하려면 `main.py`의 `USER_NAME`에 `buy_me_a_coffee`의 사용자 이름을 입력해야 합니다. 

```python
USER_NAME = 'USER_NAME' # fill the by_me_a_coffee user name
button(username=USER_NAME, floating=True, width=221)
```
