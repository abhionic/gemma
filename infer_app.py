import streamlit as st
import os; import time

st.title('gemma 2b instruction tuned')
os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle_username']
os.environ['KAGGLE_KEY'] = st.secrets['kaggle_key']
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
import keras_nlp

# initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# load the model once and use it across all users and sessions
@st.cache_resource
def load_model():
    # pp = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset('gemma_1.1_instruct_2b_en', sequence_length=128)
    return keras_nlp.models.GemmaCausalLM.from_preset('gemma_1.1_instruct_2b_en', preprocessor=None)
gemma_lm = load_model()

# react to user input
if prompt := st.chat_input('what is up?'):
    # add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    answer = gemma_lm.generate(prompt, max_length=128)
    def stream_data():
        for word in answer.split(' '):
            yield word + ' '
            time.sleep(0.02)

    # display assistant response in chat message container
    with st.chat_message('assistant'):
        response = st.write_stream(stream_data)
    # add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})
