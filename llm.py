from langchain_upstage import UpstageEmbeddings
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_upstage import ChatUpstage
from configuration import answer_examples
store ={}

def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_vector_store():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    vector_store = PineconeVectorStore.from_existing_index( embedding=embeddings, index_name='tax-index' )
    return vector_store

def get_llm():
    llm = ChatUpstage()
    return llm

def get_dictionary_chain():
    dictionary =  ["사람을 나타내는 표현 -> 거주자"] 
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
    f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 질문을 변경해주세요.
    사전 : {dictionary}
    
    
    질문 : {{question}}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

def get_qa_chain(vector_store):
    
    llm = get_llm()
    
    condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), condense_question_prompt
    )
    
    system_prompt = (
        "당신은 소득세법 전문가입니다.사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 대는 소득세법 (xx조)에 따르면 이라고 시작하면서 답변해주시고"
        "2~3 문장정도의 짧은 내용의 답변을 원합니다"
    
    "\n\n"
    "{context}"
)
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{answer}")]
        ), examples= answer_examples
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        convo_qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')
    
    return conversational_rag_chain

def get_ai_message(query):
    vector_store = get_vector_store()
    tax_chain = {"input": get_dictionary_chain()} | get_qa_chain(vector_store=vector_store)
    ai_message = tax_chain.stream({"question" : query }, config = {
        "configurable": {"session_id":"test"}
    })
    return ai_message
