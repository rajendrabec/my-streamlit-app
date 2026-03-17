import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage





st.set_page_config(page_title="Live Web Agent", layout="centered")

st.title("🌍 The Live Internet Agent")
st.write("Ask me anything about current events. I will browse the web to find the answer.")

# --- 1. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Config")
    user_api_key = st.text_input("Groq API Key:", type="password")
    st.info("Equipped with: DuckDuckGo Web Search Tool")

    use_web_search = st.toggle("Enable Web Search", value=True)
    if use_web_search:
        st.success("Web search is ON — AI will browse the internet.")
    else:
        st.warning("Web search is OFF — AI will rely on its own knowledge.")

# --- 2. THE MEMORY VAULT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. DRAW THE CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. THE CORE AGENTIC LOOP ---
if user_query := st.chat_input("Ask about today's news..."):

    if not user_api_key:
        st.error("Please enter your API Key in the sidebar.")
    else:
        # A. Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

#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
        # B. Initialize the LangGraph Agent Engine llama-3.1-8b-instant
        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=user_api_key)


        # C. THE BRIDGE: Translate Streamlit Memory -> LangGraph Memory
        langgraph_history = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                langgraph_history.append(HumanMessage(content=m["content"].encode("ascii")))
            else:
                langgraph_history.append(AIMessage(content=m["content"].encode("ascii")))


        if use_web_search:
            web_tool = DuckDuckGoSearchRun()
            tools = [web_tool]
            sys_prompt = (
                "You are a live research assistant. "
                "You MUST use the web search tool to find current information before answering. "
                "Synthesize the results cleanly."
            )
            spinner_msg = "🤖 Browsing the web and analyzing results..."
        else:
            tools = []
            sys_prompt = (
                "You are a helpful AI assistant. "
                "Answer the user's question using your own knowledge. "
                "Be concise and accurate."
            )
            spinner_msg = "🤖 Thinking..."
        # D. Execute the Agent (With a visual loading spinner)
        agent = create_react_agent(llm, tools, prompt=sys_prompt)
        with st.spinner(spinner_msg):
          # We feed the translated history into the graph
                result_state = agent.invoke({"messages": langgraph_history})

                # The final answer is the last message in the state
                bot_answer = result_state["messages"][-1].content
        with st.chat_message("assistant"):
            # Display the final answer on the UI
            st.markdown(bot_answer)
        # E. Save the final answer back to Streamlit's Vault
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
