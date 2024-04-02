from langchain_core.prompts import ChatPromptTemplate


def rag_prompt():
    # 가장 최신 버전의 프롬프트를 가져옵니다.
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    The context provided below is a transcript of the conversation in KakaoTalk chat log.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    helpful web link should be included in the answer if possible.
    User names are anonymized with asterisks.
    Answer in a bulleted form with date and time information if possible in a professional tone.
    Answer should be written in Korean.
    
    Question: 
    {question} 

    Context: 
    {context} 

    FORMAT:
    Answer in Bulleted Form(if possible)):
    - [Answer] `[Date & Time]`
    - [Answer] `[Date & Time]`
    - ...(if many)
    """
    )
    return prompt
