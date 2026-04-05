# 【最小改动 1：加在最顶部，关闭报错】
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import streamlit as st
import tempfile
import os
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.embeddings import BaichuanTextEmbeddings
import requests
import json
from langchain.agents import Tool

# ==========================
# 页面配置
# ==========================
st.set_page_config(page_title="RAG Agent Demo", layout="wide")
st.title("📚 RAG + 联网搜索 + 天气查询 Agent")

# 【最小改动 2：简单美化界面】
st.markdown("""
<style>
.stChatMessage { border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================
# 上传文件
# ==========================
uploaded_files = st.sidebar.file_uploader("上传 TXT 文档", type=["txt"], accept_multiple_files=True)
if not uploaded_files:
    st.info("📌 未上传文档，使用联网+天气模式")

# ==========================
# 文档检索器
# ==========================
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    if not uploaded_files:
        return None
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = TextLoader(temp_filepath, encoding="utf-8")
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    if not splits:
        return None

    key = st.secrets.get("BAICHUAN_API_KEY", "")
    embeddings = BaichuanTextEmbeddings(api_key=key)
    vectordb = Chroma.from_documents(splits, embeddings)
    return vectordb.as_retriever()

retriever = None
if uploaded_files:
    retriever = configure_retriever(uploaded_files)

# ==========================
# 会话消息
# ==========================
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是RAG智能助手，支持文档检索、联网搜索、天气查询"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ==========================
# 工具 1：文档检索
# ==========================
tools = []
if uploaded_files and retriever:
    tool = create_retriever_tool(
        retriever=retriever,
        name="文档检索",
        description="根据文档内容回答问题"
    )
    tools.append(tool)

# ==========================
# 工具 2：联网搜索
# ==========================
def get_search_result(question):
    from langchain_community.utilities import SerpAPIWrapper
    import traceback
    try:
        api_key = st.secrets.get("SERPAPI_KEY", "")
        if not api_key:
            return "❌ 错误：未配置 SERPAPI_KEY。请在 Streamlit Secrets 中填写。"

        search = SerpAPIWrapper(serpapi_api_key=api_key)
        result = search.run(question)
        return result
    except Exception as e:
        st.error(f"🔴 搜索失败详情: {str(e)}")
        return f"❌ 联网搜索失败：{str(e)} \\n\\n建议：检查网络连接或稍后重试。"

searchTool = Tool(
    name="get_search_result",
    description="联网获取实时信息、新闻、知识",
    func=get_search_result
)

# ==========================
# 工具 3：天气查询
# ==========================
def get_weather(loc):
    try:
        loc = loc.replace("天气", "").replace("市", "").replace("省", "").strip()
        api_key = st.secrets.get("WEATHER_API_KEY", "")
        url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={loc}&language=zh-Hans&unit=c"
        return json.dumps(requests.get(url).json(), ensure_ascii=False)
    except:
        return "天气查询失败"

weatherTool = Tool(
    name="get_weather",
    description="查询城市天气，输入：北京、上海、佛山",
    func=get_weather
)

tools.extend([searchTool, weatherTool])

# ==========================
# 记忆 + 提示词
# ==========================
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history",
                                  output_key="output")

instructions = """
你是具备文档检索、联网搜索、天气查询能力的智能助手。
有文档优先查文档；无文档可联网；问天气直接调用天气工具。
不知道就说不知道，不编造。
"""

base_prompt_template = """
{instructions}

TOOLS:
------

You have access to the following tools:
{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(base_prompt_template).partial(instructions=instructions)

# ==========================
# LLM + Agent
# ==========================
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=st.secrets.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
    temperature=0
)

agent = create_react_agent(llm, tools, prompt)
# 【最小改动3：verbose=False → 关闭思考打印】
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, handle_parsing_errors=True,
                               max_iterations=5)

# ==========================
# 聊天交互
# ==========================
user_query = st.chat_input("输入问题...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # 【最小改动4：去掉回调，界面干净】
        # st_cb = StreamlitCallbackHandler(st.container())
        # config = {"callbacks": [st_cb]}

      try:
            # 打开回调，显示工具调用
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            config = {"callbacks": [st_cb]}
            
            res = agent_executor.invoke({"input": user_query}, config=config)
            ans = res["output"]
        except Exception as e:
            ans = f"我无法回答：{user_query}"
    
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.write(ans)
