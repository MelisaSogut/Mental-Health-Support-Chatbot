import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Settings ---
VECTOR_DB_PATH = "./chroma_db"

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Mode
st.markdown("""
    <style>
    /* Sakura Theme - Lighter Pink */
    .main {
        background-color: #2d1520;
        color: #fafafa;
    }
    
    .stApp {
        background-color: #2d1520;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #3d2030;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #fafafa;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #3d3d3d;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4a9eff;
    }
    
    /* Chat input */
    .stChatInput > div > div > textarea {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #3d3d3d;
    }
    
    /* Messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .user-message {
        background: linear-gradient(135deg, #be185d 0%, #f472b6 100%);
        border-left: 4px solid #f9a8d4;
        color: #ffffff;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #3d2838 0%, #5a3a4d 100%);
        border-left: 4px solid #fbcfe8;
        color: #fce7f3;
        line-height: 1.7;
    }
    
    .bot-message p {
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    
    .message-header {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .message-content {
        font-size: 1rem;
        line-height: 1.8;
        color: #fafafa;
    }
    
    /* Buttons - Lighter Sakura Pink */
    .stButton > button {
        background-color: #ec4899;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #db2777;
    }
    
    /* Markdown text */
    .markdown-text-container {
        color: #fafafa;
    }
    
    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Spinner - Lighter Sakura */
    .stSpinner > div {
        border-top-color: #f472b6 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1a4d2e;
        color: #68d391;
    }
    
    .stError {
        background-color: #4a1a1a;
        color: #fc8181;
    }
    
    /* Divider */
    hr {
        border-color: #3d3d3d;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #a0aec0;
        font-size: 0.9rem;
        padding: 1rem;
        border-top: 1px solid #3d3d3d;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

def clean_text(text):
    """Remove [INST] and </s> tags and clean the response"""
    # Remove [INST] ... [/INST] tags
    text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    # Remove <s> and </s> tags
    text = re.sub(r'</?s>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components"""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found in .env file!")
            return None
        
        # Load embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Load Chroma database
        vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Initialize Gemini model with custom prompt
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {str(e)}")
        return None

def enhance_response_with_gemini(response, query):
    """Enhance and format the response using Gemini"""
    try:
        load_dotenv()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.4
        )
        
        prompt = f"""You are a compassionate mental health assistant. 

Based on this information: {response}

And this user question: {query}

Provide a clear, empathetic, and well-structured response. Format your response with:
- Clear paragraphs
- Use emoji sparingly for emphasis (1-2 maximum)
- Be supportive and professional
- Keep it concise but informative
- Use simple language

Do not include any source references, instructions, or meta-information. Just provide the helpful response."""

        enhanced = llm.invoke(prompt)
        return enhanced.content
        
    except Exception as e:
        return response

def display_message(role, content):
    """Display a chat message"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">👤 You</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Clean the content
        cleaned_content = clean_text(content)
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-header">🧠 Mental Health Assistant</div>
            <div class="message-content">{cleaned_content}</div>
        </div>
        """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧠 Mental Health Support")
    st.markdown("---")
    
    st.markdown("""
    ### About
    This chatbot provides information and support regarding mental health conditions and advice.
    
    ### How to use:
    1. Type your question in the chat box
    2. Press Enter or click Send
    3. Review the compassionate response
    
    ### Important Note:
    ⚠️ This is an informational tool only. For professional medical advice, please consult a licensed healthcare provider.
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Model:** Google Gemini 2.0 Flash")
    st.markdown("**Dataset:** Mental Health Dataset")

# Main content
st.title("💬 Mental Health Support Chatbot")
st.markdown("Ask me anything about mental health conditions, symptoms, and advice.")
st.markdown("---")

# Initialize chatbot
if not st.session_state.initialized:
    with st.spinner("🔄 Initializing chatbot... Please wait."):
        st.session_state.qa_chain = initialize_chatbot()
        if st.session_state.qa_chain:
            st.session_state.initialized = True
            st.success("✅ Chatbot initialized successfully!")
        else:
            st.error("❌ Failed to initialize chatbot. Please check your configuration.")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    display_message(message["role"], message["content"])

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    display_message("user", user_input)
    
    # Get bot response
    with st.spinner("🤔 Thinking..."):
        try:
            result = st.session_state.qa_chain({"query": user_input})
            raw_response = result["result"]
            
            # Clean and enhance the response
            cleaned_response = clean_text(raw_response)
            
            # Enhance with Gemini for better formatting
            enhanced_response = enhance_response_with_gemini(cleaned_response, user_input)
            
            # Add bot message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": enhanced_response
            })
            
            # Display bot message
            display_message("assistant", enhanced_response)
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
    <p>🌟 Remember: You're not alone. Seeking help is a sign of strength.</p>
    <p>If you're experiencing a mental health emergency, please contact emergency services or a crisis hotline.</p>
</div>

""", unsafe_allow_html=True)
