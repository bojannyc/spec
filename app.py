import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate


st.set_page_config(page_title="Demandey User Story Assistant", page_icon="🌍", layout="centered")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Demandey User Story Assistant")
st.markdown("This time next we'll be milioners!")


class StreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container, textType):
        self.container = container
        self.text=""
        self.textType = textType
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.textType == "text" :
            self.text+=token
            self.container.markdown(self.text) 
        elif self.textType == "code" :
            self.text+=token
            self.container.code(self.text) 

# App framework
with st.form("my_form"):
    gpt_model = st.selectbox('Select GPT Model', ['gpt-4-turbo-preview'])
    temperature = st.slider('Select GenAI temperature', 0.1,1.0,0.3,0.1,)
    st.divider()
    app_type = st.selectbox('Select Story Type', ['Web App', 'Mobile App', 'Backend'])
    st.divider()
    title = st.text_input('User Story Title')
    businessFunctionality = st.text_area('Describe functionality')
    data = st.text_area('List data elements, comma separated')
    other = st.text_area('Describe security requirements, navigation flow, etc.')
    submitted = st.form_submit_button("Submit")

if title and businessFunctionality and data and submitted:
    if app_type=='Mobile App':
        prompt_template = PromptTemplate(
        input_variables = ['title', 'businessFunctionality','data','other'], 
        template="""Act as a software architect. 
        Your task is to create User Story for MOBILE application for developer, by using following inputs:
        Leverage the following inputs:
        User Story Tytle: {title}
        Business Functionality of the Web app/module:{businessFunctionality}
        Data Elements: {data}
        Other Aspects (security requirements, navigation flow, etc.) : {other}

        The output MUST be CONSISTENT with following format:
        User Roles: Define the different user roles that will interact with the app (e.g., end-user, admin).
        Functional Requirements: List the core functionalities the app must provide (e.g., user authentication, data display).
        Non-Functional Requirements: Specify performance metrics, security requirements, and usability goals.
        Navigation Flow: Outline the navigation structure, including the sequence of screens and user interactions.
        Design Elements: Define the visual elements such as themes, colors, and font styles.
        Device Compatibility: Specify which devices and OS versions the app should be compatible with.
        Accessibility Requirements: Include guidelines for making the app accessible to all users, including those with disabilities.
        """
        )
    elif (app_type=='Web App'):
        prompt_template = PromptTemplate(
        input_variables = ['title', 'businessFunctionality','data','other'], 
        template="""Act as a software architect. 
        Your task is to create User Story for WEB application for developer, by using following inputs:
        Leverage the following inputs:
        User Story Tytle: {title}
        Business Functionality of the Web app/module:{businessFunctionality}
        Data Elements: {data}
        Other Aspects (security requirements, navigation flow, etc.) : {other}

        The output MUST be CONSISTENT with following format:
        User Roles: Similar to mobile apps, define the different user roles for the web application.
        Functional Requirements: Describe the functionalities specific to web usage (e.g., cross-browser support, responsive design).
        Non-Functional Requirements: Highlight security measures, load times, and scalability.
        Navigation & Flow: Detail the web app's architecture, including page hierarchy and user pathways.
        Design Elements: Specify the design elements tailored to web experiences.
        Accessibility Standards: Ensure compliance with web accessibility standards (e.g., WCAG).
        SEO Considerations: Include parameters for SEO optimization, such as metadata and URL structure.
        """
        )
    else:
        prompt_template = PromptTemplate(
        input_variables = ['title', 'businessFunctionality','data','other'], 
        template="""Act as a software architect. 
        Your task is to create User Story for Nodejs application for developer, by using following inputs:
        Leverage the following inputs:
        User Story Tytle: {title}
        Business Functionality of the Web app/module:{businessFunctionality}
        Data Elements: {data}
        Other Aspects (security requirements, navigation flow, etc.) : {other}

        The output MUST be CONSISTENT with following format:
        Data Models: Define the structure of the database models and relationships.
        API Endpoints: List the API endpoints that will be needed, along with their methods (GET, POST, etc.), inputs, and expected outputs.
        Authentication & Authorization: Describe the mechanisms for user authentication and authorization levels.
        Business Logic: Outline the core logic that will process the data and handle requests.
        Error Handling: Define the approach for managing errors and exceptions.
        Performance Metrics: Set benchmarks for response times and throughput.
        Security Measures: Include parameters for data encryption, secure access, and other security protocols.
        """
        )

    t1=st.empty()
    stream_handler1 = StreamHandler(t1, "text") 
    llm=ChatOpenAI(streaming=True,model=gpt_model, callbacks=[stream_handler1],temperature=temperature)
    title_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, output_key='result')

    
    result = title_chain.run(title=title, businessFunctionality=businessFunctionality, data=data, other=other)
