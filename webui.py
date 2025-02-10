import streamlit as st
import requests
import time
from dotenv import load_dotenv
import os
from ResearchAgent import *
from llm_util import *


def init_session_state():
    """初始化会话状态"""
    if "results" not in st.session_state:
        st.session_state.results = []
    if "scores" not in st.session_state:
        st.session_state.scores = []
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "search"
    if "search_completed" not in st.session_state:
        st.session_state.search_completed = False


def search_papers(model: str, user_input: str, num_each_query: int = 5):
    """调用ResearchAgent进行论文搜索"""
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.text("开始搜索和总结。")

    try:
        # 初始化LLM客户端和ResearchAgent
        llm = LLM_client(model)
        agent = ResearchAgent(llm)

        criteria = agent.analyze_query(user_input)
        criteria.max_results = num_each_query

        progress_bar.progress(0.2)
        status_text.text(f"提取查询关键词:{criteria.keywords}")

        results = []
        for keyword in criteria.keywords:
            query = keyword
            result = get_arxiv_results(query, max_results=criteria.max_results)
            results.extend(result)

        num_first_papers = len(results)

        progress_bar.progress(0.4)
        status_text.text("检索Arxiv相关论文。")

        progress_bar.progress(0.6)
        status_text.text(f"对{num_first_papers}篇论文进行筛选和排序。")

        # 重新排序论文
        papers, scores = agent._rerank_results(user_input, results)

        progress_bar.progress(0.8)
        status_text.text("生成阅读总结。")

        messages = []
        for paper in papers:
            message = get_arxiv_message(paper)
            messages.append(message)
        summary = agent._generate_summary(messages, scores)

        # 更新进度
        progress_bar.progress(1.0)
        status_text.text("搜索完成！")

        return {"results": papers, "scores": scores, "summary": summary}

    except Exception as e:
        st.error(f"搜索失败: {str(e)}")
        return None


def show_search_page():
    """显示搜索页面"""
    st.title("PaperCreeper: 论文检索助手")
    st.markdown(
        """
## 🎯主要功能介绍

- 支持自然语言描述的查询，并根据查询自动生成若干检索词，多次迭代、优化搜索结果。
- 支持对检索结果进行评估和重新排序，提升检索质量。
- 支持对检索论文进行阅读和总结，并生成独特见解，方便用户快速聚焦有用的论文。
"""
    )

    # 模型选择
    st.header("🤖 请选择LLM模型")
    model = st.selectbox(
        "选择模型",
        [
            "openai/gpt-4o-2024-11-20",
            "openai/gpt-4o-mini-2024-07-18",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-chat",
        ],
        label_visibility="collapsed",
    )

    # 查询输入
    st.header("🔍 请输入查询")
    st.markdown(
        """
    请输入您的论文检索需求 (中文或英文)，可以包含以下信息：
    - 研究主题或关键词
    - 发表年份要求（可选）
    - 研究领域（可选）
    
    例如：
    "Find the latest research on multi-agents for large language models."
    """
    )
    query = st.text_area("输入查询", height=100, label_visibility="collapsed")

    # 处理过程
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("开始搜索", type="primary", use_container_width=True)

    if search_button:
        # 执行搜索
        result = search_papers(model, query)

        if result:
            st.session_state.results = result["results"]
            st.session_state.scores = result["scores"]
            st.session_state.summary = result["summary"]
            st.session_state.search_completed = True
            st.session_state.current_page = "results"
            st.rerun()


def show_results_page():
    """显示结果页面"""
    st.title("检索结果")

    # 返回搜索按钮
    if st.button("返回搜索", type="secondary"):
        st.session_state.current_page = "search"
        st.session_state.search_completed = False
        st.rerun()

    # 检索结果
    st.header("文献检索结果")
    if st.session_state.results:
        num_papers = len(st.session_state.results)  # results包含论文列表
        st.markdown(f"一共检索得到了{num_papers}篇相关论文。")

        # 创建表格标题
        table_header = "| 序号 | 标题 | 年份 | 作者 | 相关度评分 | ArXiv链接 |\n| --- | --- | --- | --- | --- | --- |"
        table_rows = []

        # 为每篇论文创建表格行
        for i, (result, score) in enumerate(
            zip(st.session_state.results, st.session_state.scores), 1
        ):
            authors = ", ".join([author.name for author in result.authors])
            row = f"| {i} | {result.title} | {result.published.year} | {authors} | {score:.1f}/10 | [链接]({result.entry_id}) |"
            table_rows.append(row)

        # 组合表格
        table = "\n".join([table_header] + table_rows)
        st.markdown(table)

    # 总结结果
    st.header("文献阅读总结结果")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)


def main():
    load_dotenv()

    # 初始化会话状态
    init_session_state()

    # 根据当前页面状态显示不同的页面
    if st.session_state.current_page == "search":
        show_search_page()
    else:
        show_results_page()


if __name__ == "__main__":
    main()
