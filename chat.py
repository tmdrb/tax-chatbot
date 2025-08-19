from llm import get_ai_message
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title='소득세 챗봇')

st.title("소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다.")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []


for dic in st.session_state.message_list:
    with st.chat_message(dic['role']):
            st.write(dic['content'])
   
            
if user_question := st.chat_input(placeholder='소득세에 관련된 궁금한것들을 말해주세요'):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':"user","content":user_question})

    with st.spinner("답변을 생성하는 중입니다."):
        with st.chat_message('ai'):
            
            ai_message = st.write_stream(get_ai_message(user_question))
            st.session_state.message_list.append({'role':"ai","content":ai_message})
