import os
# --- 1. 配置镜像源 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 2. 页面设置 (改为侧边栏导航风格) ---
st.set_page_config(
    page_title="多领域知识分类检索系统",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. 核心逻辑：自动分类与索引 ---
@st.cache_resource
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 读取 data 目录下所有文件
    loader = DirectoryLoader('data/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, {}

    # --- 关键修改：自动打标签 (Text Classification 模拟) ---
    categorized_docs = []
    categories = set()
    
    # 1. 定义新的关键词列表 (对应生成数据的三个类别)
    # AI 保持不变
    ai_keywords = ['learning', 'neural', 'intelligence', 'gpt', 'python', 'data', 'cloud']
    
    # FinTech (金融科技) - 替代原来的 geo
    fintech_keywords = ['blockchain', 'bitcoin', 'payment', 'finance', 'wallet', 'economy', 'bank']
    
    # Humanities (人文常识) - 替代原来的 sci
    humanities_keywords = ['history', 'culture', 'art', 'philosophy', 'literature', 'civilization', 'museum']
    
    for doc in raw_docs:
        filename = doc.metadata['source'].lower()
        content = doc.page_content.lower()
        
        # 默认分类
        category = "其他资讯 (General)"
        
        # 根据文件名或内容判断分类
        if any(k in filename or k in content for k in ai_keywords):
            category = "🤖 AI与前沿技术"
        elif any(k in filename or k in content for k in fintech_keywords):
            category = "💰 金融科技观察"
        elif any(k in filename or k in content for k in humanities_keywords):
            category = "📚 人文历史常识"
            
        # 将分类写入 metadata
        doc.metadata['category'] = category
        categorized_docs.append(doc)
        categories.add(category)

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(categorized_docs)
    
    # 建立向量索引
    vector_db = FAISS.from_documents(splits, embeddings)
    
    return vector_db, raw_docs, list(categories)

# --- 4. 初始化 ---
with st.spinner("正在加载分类模型与知识库..."):
    vector_db, raw_docs, category_list = initialize_system()

# --- 5. UI 布局：左侧筛选，右侧检索 ---
# Topic 2 要求：Classification labels as filters 

with st.sidebar:
    st.header("📂 领域导航")
    st.markdown("请选择要检索的知识领域：")
    
    # 添加“全部”选项
    selected_category = st.radio(
        "选择分类 (Topic Filter):",
        ["🌐 全部领域 (All Topics)"] + sorted(list(category_list))
    )
    
    st.markdown("---")
    st.info(f"📚 当前库中文档总数: {len(raw_docs)}")
    if selected_category != "🌐 全部领域 (All Topics)":
        # 统计当前分类下的文档数
        count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)
        st.success(f"当前分类包含文档: {count} 篇")

# 主界面
st.title("📑 Topic-Filtered Retrieval System")
st.caption("基于文本分类的定向检索系统 | Topic 2 Implementation")

# 搜索区
query = st.text_input("在该领域内搜索关键词...", placeholder="输入查询内容...")
search_btn = st.button("🔍 检索文档", type="primary")

if (query or search_btn) and vector_db:
    # --- 检索逻辑 ---
    # 1. 先进行向量检索 (召回 Top 10，多拿一点方便后面过滤)
    results = vector_db.similarity_search(query, k=15)
    
    # 2. 后置过滤 (Post-filtering)：只保留用户选中分类的结果
    if selected_category != "🌐 全部领域 (All Topics)":
        filtered_results = [doc for doc in results if doc.metadata.get('category') == selected_category]
    else:
        filtered_results = results

    # 取前 4 个展示
    final_results = filtered_results[:4]

    st.markdown(f"### 🔎 '{selected_category}' 领域下的检索结果")
    
    if not final_results:
        st.warning(f"在 '{selected_category}' 分类下未找到相关文档。")
    else:
        for doc in final_results:
            cat_tag = doc.metadata.get('category')
            source_name = doc.metadata['source'].split('/')[-1]
            
            # 使用 Streamlit 的 expander 样式展示，看起来像文件列表
            with st.expander(f"📄 {source_name}  [{cat_tag}]", expanded=True):
                st.markdown(f"**...{doc.page_content}...**")
                st.caption(f"来源: {doc.metadata['source']}")

elif not vector_db:
    st.error("未找到数据，请检查 data 文件夹。")