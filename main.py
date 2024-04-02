import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    ConfigurableField,
)
from langchain_community.document_transformers import LongContextReorder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
import kakaotalk_loader as kakao
import prompt as prmpt
import embeddings
import retriever
import tempfile
from utils import print_messages, StreamHandler

st.set_page_config(page_title="카톡GPT", page_icon="💬")
st.title("카톡GPT💬")
st.markdown(
    """by [테디노트](https://www.youtube.com/c/teddynote). 소스코드 활용시 반드시 **출처**를 밝혀주세요🙏"""
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    openai_api_key = st.text_input(
        "🔑 OpenAI API 키",
        type="password",
    )
    if openai_api_key:
        st.session_state["OPENAI_API_KEY"] = openai_api_key
    st.markdown(
        "📌 CSV파일 다운로드 방법\n\n`채팅방`-`우측상단 햄버거메뉴`-`채팅방 설정`-`대화 내용 관리`-`대화 내용 저장`"
    )
    kakaotalk_file = st.file_uploader("📄 카톡 CSV 파일 업로드", type=["csv"])
    if kakaotalk_file:
        if "OPENAI_API_KEY" not in st.session_state:
            st.info("OpenAI API Key를 입력해 주세요.")
        else:
            st.session_state["kakaotalk_file"] = kakaotalk_file

if "kakaotalk_file" in st.session_state and "retriever" not in st.session_state:
    with st.sidebar:
        with st.status("파일을 처리 중입니다 🧑‍💻👩‍💻", expanded=True) as status:
            with tempfile.NamedTemporaryFile() as f:
                f.write(st.session_state["kakaotalk_file"].read())
                f.flush()
                # 카카오톡 로더
                loader = kakao.KaKaoTalkLoader(f.name, encoding="utf8")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=0
                )
                documents = loader.load_and_split(text_splitter=text_splitter)
                st.write("① 임베딩 생성")
                status.update(label="① 임베딩을 생성 중..🔥", state="running")
                # Embedding 생성
                embeddings = embeddings.embedding_factory(
                    api_key=st.session_state["OPENAI_API_KEY"]
                )

                st.write("② DB 인덱싱")
                status.update(label="② DB 인덱싱 생성 중..🔥", state="running")
                # VectorStore 생성
                faiss = FAISS.from_documents(documents, embeddings["faiss"])
                chroma = Chroma.from_documents(documents, embeddings["chroma"])

                st.write("③ Retriever 생성")
                status.update(label="③ Retriever 생성 중..🔥", state="running")
                # FAISSRetriever 생성
                faiss_retriever = retriever.FAISSRetrieverFactory(faiss).create(
                    search_kwargs={"k": 30},
                )

                # SelfQueryRetriever 생성
                self_query_retriever = retriever.SelfQueryRetrieverFactory(
                    chroma
                ).create(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    api_key=st.session_state["OPENAI_API_KEY"],
                    search_kwargs={"k": 30},
                )

                # 앙상블 retriever를 초기화합니다.
                ensemble_retriever = retriever.EnsembleRetrieverFactory(None).create(
                    retrievers=[faiss_retriever, self_query_retriever],
                    weights=[0.4, 0.6],
                )
                reordering = LongContextReorder()

                combined_retriever = ensemble_retriever | RunnableLambda(
                    reordering.transform_documents
                )
                st.session_state["retriever"] = combined_retriever
                st.write("완료 ✅")
                status.update(label="완료 ✅", state="complete", expanded=False)
        st.markdown(f'💬 `{st.session_state["kakaotalk_file"].name}`')
        st.markdown(
            "🔔참고\n\n**새로운 카톡 파일** 로 대화를 시작하려면, `새로고침` 후 진행해 주세요"
        )


# 이전 대화기록을 출력해 주는 코드
print_messages()


if user_input := st.chat_input("메시지를 입력해 주세요."):
    if "OPENAI_API_KEY" not in st.session_state:
        st.info("OpenAI API Key를 입력해 주세요.")
    elif "retriever" not in st.session_state:
        st.info("KakaoTalk CSV 파일을 업로드해 주세요.")
    else:
        # 사용자가 입력한 내용
        st.chat_message("user").write(f"{user_input}")
        st.session_state["messages"].append(
            ChatMessage(role="user", content=user_input)
        )

        # AI의 답변
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())

            llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                temperature=0,
                streaming=True,
                callbacks=[stream_handler],
                api_key=st.session_state["OPENAI_API_KEY"],
            ).configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key="gpt4",
                gpt3=ChatOpenAI(
                    model="gpt-3-turbo",
                    temperature=0,
                    streaming=True,
                    callbacks=[stream_handler],
                    api_key=st.session_state["OPENAI_API_KEY"],
                ),
            )

            chain = (
                {
                    "context": st.session_state["retriever"],
                    "question": RunnablePassthrough(),
                }
                | prmpt.rag_prompt()
                | llm
                | StrOutputParser()
            )

            response = chain.invoke(
                user_input,
            )
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
