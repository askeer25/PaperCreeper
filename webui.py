import streamlit as st
from dotenv import load_dotenv
from ResearchAgent import *
from llm_util import *
import asyncio

# Page config
st.set_page_config(
    page_title="PaperCreeper",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """

    """
)


def init_session_state():
    """初始化会话状态"""
    if "papers" not in st.session_state:
        st.session_state.papers = []
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


async def show_search_page(agent: ResearchAgent):
    # Add a decorative header with emoji and custom styling
    st.markdown(
        """
        <h1 style='text-align: center; color: #2E4053;'>
            🔍 PaperCreeper - 论文助手
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Add a subtle divider
    st.markdown("---")

    # Create a clean container for the search input
    with st.container():
        st.markdown(
            """
            <h3 style='color: #566573;'>您想探索什么领域？</h3>
            """,
            unsafe_allow_html=True,
        )

        query = st.text_area(
            label="请输入您想要查询的主题",
            height=100,
            placeholder="例如：最新的深度学习研究进展...",
        )

    # Add styled suggestions
    # st.markdown(
    #     """
    #     <div style='background-color: #F8F9F9; padding: 20px; border-radius: 10px;'>
    #     <h4 style='color: #566573;'>💡 建议的提问方式</h4>
    #     <ul style='color: #626567;'>
    #         <li>查找一篇人工智能领域的论文</li>
    #         <li>了解一下深度学习的最新进展</li>
    #         <li>有关自然语言处理的研究</li>
    #     </ul>
    #     </div>
    # """,
    #     unsafe_allow_html=True,
    # )

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Create a centered search button with custom styling
    _, middle, _ = st.columns([1, 2, 1])  # Adjusted column ratios for better centering
    with middle:
        search_button = st.button(
            "🔎 开始搜索",
            type="primary",
            use_container_width=True,  # Make button stretch to container width
            key="search_main",
        )

    if search_button:
        # Reset processing state when returning from results page
        st.session_state.processing = False
        if not st.session_state.processing:
            st.session_state.processing = True
            with st.spinner("🔍 正在为您检索相关论文..."):
                results = await agent._search_arxiv(query)  # 异步搜索

                if results:
                    st.session_state.current_page = "results"
                    st.session_state.papers = results["papers"]
                    st.session_state.scores = results["scores"]
                    st.session_state.search_completed = True
                    st.rerun()


def show_results_page(agent: ResearchAgent):
    """显示结果页面"""
    st.markdown(
        """
        <h1 style='text-align: center; color: #2E4053;'>
            📇 检索结果
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # 初始化paper_summaries（如果不存在）
    if "paper_summaries" not in st.session_state:
        st.session_state.paper_summaries = {}

    # 检索结果
    if st.session_state.papers:
        # 显示检索到的论文
        for i, paper in enumerate(st.session_state.papers):
            with st.container():
                # 使用更现代的卡片样式
                st.markdown(
                    f"""
                    <div style='background-color: #FFFFFF; 
                              padding: 25px; 
                              border-radius: 15px; 
                              margin: 15px 0;
                              box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
                    <h3 style='color: #2E4053; margin-bottom: 15px;'>{i+1}. {paper.title}</h3>
                    """,
                    unsafe_allow_html=True,
                )

                # 优化相关度显示
                score = st.session_state.scores[i]
                star_count = min(5, max(1, round((score / 10) * 5)))
                stars = "⭐" * star_count

                # 改进列布局
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**相关度:** {stars}")
                    st.markdown(f"**发布日期:** {paper.published.strftime('%Y-%m-%d')}")
                with col2:
                    authors = [author.name for author in paper.authors]
                    st.markdown(
                        f"**作者:** {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}"
                    )
                with col3:
                    # 操作按钮垂直排列
                    st.button(
                        "🤖 AI总结",
                        key=f"summary_button_{i}",
                        type="primary",
                        use_container_width=True,
                        on_click=lambda paper=paper, i=i: generate_summary(
                            agent, paper, i
                        ),
                    )

                    st.markdown(
                        f"""
                        <a href="{paper.entry_id}" target="_blank" 
                           style="text-decoration: none; 
                                  background-color: #3498DB; 
                                  color: white; 
                                  padding: 8px 15px; 
                                  border-radius: 5px; 
                                  width: 100%;
                                  text-align: center;
                                  display: inline-block;
                                  margin-top: 5px;">
                            📎 查看原文
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )

                # 改进摘要显示
                with st.expander("📄 查看摘要", expanded=False):
                    st.markdown(f"{paper.summary}")

                # 显示AI总结（如果有）
                if i in st.session_state.paper_summaries:
                    with st.expander("🤖 AI智能解读", expanded=True):
                        st.markdown(st.session_state.paper_summaries[i])

                st.markdown("</div>", unsafe_allow_html=True)

        # 底部控制区
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("← 返回搜索", type="secondary", use_container_width=True):
                st.session_state.current_page = "search"
                st.session_state.search_completed = False
                st.rerun()

        with col2:
            if st.button(
                "✨ 生成整体总结",
                key="global_summary",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("🎯 正在生成研究领域的整体分析..."):
                    st.session_state.summary = agent._summarize_papers(
                        st.session_state.papers, st.session_state.scores
                    )

        # 显示整体总结
        if st.session_state.summary:
            st.markdown(
                """
                <div style='background-color: #F7F9FA; 
                          padding: 20px; 
                          border-radius: 10px; 
                          margin-top: 20px;
                          border-left: 5px solid #3498DB;'>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state.summary)
            st.markdown("</div>", unsafe_allow_html=True)


def generate_summary(agent: ResearchAgent, paper, index: int):
    """生成单篇论文的摘要"""
    if index not in st.session_state.paper_summaries:
        with st.spinner("正在总结论文..."):
            summary = agent._summarize_paper(paper)
            st.session_state.paper_summaries[index] = summary


async def main():
    # Load environment variables first
    load_dotenv()

    try:
        # Initialize LLM client and ResearchAgent
        llm = LLM_client("openai/gpt-4o-2024-11-20")
        agent = ResearchAgent(llm)

        # Initialize session state
        init_session_state()

        # Show different pages based on current state
        if st.session_state.current_page == "search":
            await show_search_page(agent)
        elif st.session_state.current_page == "results":
            show_results_page(agent)
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
