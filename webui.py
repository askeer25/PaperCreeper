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

    # 重置summary
    st.session_state.paper_summaries = {}

    # 检索结果
    if st.session_state.papers:
        # 显示检索到的论文
        for i, paper in enumerate(st.session_state.papers):
            with st.container():
                st.markdown(
                    f"""
                    <div style='background-color: #F8F9F9; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <h3 style='color: #2E4053;'>{i+1}. {paper.title}</h3>
                    """,
                    unsafe_allow_html=True,
                )

                # Calculate stars (5 stars maximum for score of 10)
                star_count = round((st.session_state.scores[i] / 10) * 5)
                stars = "⭐" * star_count

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**相关度:** {stars}")
                    st.markdown(
                        f"**作者:** {', '.join([author.name for author in paper.authors])}"
                    )
                    st.markdown(f"**发布日期:** {paper.published.year}")

                with col2:
                    # 为每篇论文添加总结按钮
                    if st.button(
                        f"🤖 AI总结", key=f"summary_button_{i}", type="primary"
                    ):
                        with st.spinner("正在总结论文..."):
                            summary = agent._summarize_paper(paper)
                            st.session_state.paper_summaries[i] = summary

                # Create an expandable section for abstract
                with st.expander("📄 查看摘要"):
                    st.markdown(f"{paper.summary}")

                # Style the link with a button-like appearance
                st.markdown(
                    f"""
                    <a href="{paper.entry_id}" target="_blank" 
                       style="text-decoration: none; 
                              background-color: #3498DB; 
                              color: white; 
                              padding: 8px 15px; 
                              border-radius: 5px; 
                              font-size: 14px;">
                        📎 查看原文
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

                # Display AI summary in a nicer card if available
                if i in st.session_state.paper_summaries:
                    st.markdown(
                        """
                        <div style='background-color: #E8F6F3; 
                                  padding: 20px; 
                                  border-radius: 10px; 
                                  margin: 15px 0;
                                  border: 1px solid #A3E4D7;
                                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <h4 style='color: #117A65; margin-bottom: 10px;'>
                            🤖 AI 智能解读
                        </h4>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(st.session_state.paper_summaries[i])
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        # 创建三列布局
        _, middle, _ = st.columns([1, 2, 1])
        with middle:
            if st.button(
                "✨ 生成整体总结",
                key="global_summary",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("🎯 正在生成研究领域的整体分析..."):
                    summary = agent._summarize_papers(
                        st.session_state.papers, st.session_state.scores
                    )
                    st.session_state.summary = summary

        # 显示总结内容，使用更优雅的卡片设计
        if st.session_state.summary:
            # st.markdown(
            #     """
            # <div style='background-color: #E8F6F3;
            #       padding: 25px;
            #       border-radius: 15px;
            #       box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            #       margin: 20px 0;
            #       border: 1px solid #A3E4D7;'>
            #     <h3 style='color: #117A65;
            #          margin-bottom: 20px;
            #          border-bottom: 2px solid #A3E4D7;
            #          padding-bottom: 10px;'>
            #     📊 研究领域综述
            #     </h3>
            # """,
            #     unsafe_allow_html=True,
            # )
            st.markdown(st.session_state.summary)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # 优化返回按钮的样式和位置
        st.markdown("<br>", unsafe_allow_html=True)
        left_col, _, _ = st.columns([1, 1, 1])
        with left_col:
            if st.button("← 返回搜索", type="secondary", use_container_width=True):
                st.session_state.current_page = "search"
                st.session_state.search_completed = False
                st.rerun()


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
