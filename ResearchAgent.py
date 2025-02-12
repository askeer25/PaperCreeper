import json
from typing import List, Dict
from arxiv_util import *  # Ensure these functions are defined
from llm_util import *
from dotenv import load_dotenv
import logging
import asyncio


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ResearchAgent:
    def __init__(self, llm_client: LLM_client):
        self.llm = llm_client

        self.KEYWORD_PROMPT = """You are an academic search assistant, your task is to generate queries to retrieve relevant papers on arxiv.If the user enters Chinese, please translate it into English first and ensure professionalism and accuracy. More often than not, a user will ask a basic question that they wish to learn more about, however it might not be in the best format. 

User input: {query}

<queries>["query_1", "query_2", ...]</queries>

"""
        self.SUMMARY_PAPER_PROMPT = """您是一位学术研究助手，请对以下论文提供详细总结与分析。

论文标题: {paper}, 论文摘要: {abstract}

您的总结应涵盖以下方面：

1. 主要研究方向与贡献：描述论文的核心研究领域、目标及主要贡献。作者试图解决什么问题？他们采用了何种方法来解决问题？
2. 创新点与关键发现：突出论文中的新颖方法、技术或研究成果，这些内容如何使该论文区别于其他相关研究？其方法或结果的独特之处在哪里？
3. 论文间的联系与关系：识别这些论文之间的关联性，例如它们是否探讨了相似的问题、使用了互补的方法，或者在彼此的基础上进行了扩展。是否存在贯穿这些论文的总体趋势或共同主题？
4. 对广泛领域的相关性：简要讨论这些论文在其各自领域中的意义

请使用markdown语法回答问题，只可以使用列表、文本、加粗字体，不要使用其他格式。
请以中文作答, 并确保您的回答专业准确, 不要超过300字
"""

        self.SUMMARY_PROMPT = """您是一位专业的学术研究助手，请根据提供的相关性评分，对这组论文进行全面的分析和总结。

    论文集合: {papers}
    相关性评分：{scores}

    请从以下几个方面进行整体分析：

    1. 研究主题概览
    - 概括这组论文涉及的主要研究领域和核心主题
    - 分析研究主题的分布和侧重点

    2. 研究方法与技术路线
    - 总结这组论文采用的主要研究方法
    - 比较不同论文间的技术路线异同
    - 分析方法论的演进趋势

    3. 关键发现与贡献
    - 提炼最重要的研究发现和突破
    - 评估各项发现的创新性和影响力
    - 根据相关性评分分析研究成果的重要程度

    4. 研究脉络与趋势
    - 分析论文之间的承接关系和演进路径
    - 识别该领域的发展趋势和未来方向
    - 指出潜在的研究空白和机会

    请使用简洁专业的语言，确保分析客观全面，突出重点发现。
    总结长度控制在600字以内。结尾请附上论文的ArXiv链接。
    """

        self.RERANK_PROPMT = """
You are an academic relevance scorer that rates papers on a scale from 0-10 based on relevance to user interest. 

User input:
{input}

Consider this paper:
{paper_info}

Rate how relevant this paper is on a scale of 0-10, where:
0 = Not relevant at all
5 = Moderately relevant 
10 = Extremely relevant

Provide ONLY a single number score between 0-10 with no other text.
"""

    def _extract_keywords(self, user_input: str) -> List[str]:
        logging.info("从用户输入中提取关键词.")
        messages = [
            {"role": "system", "content": "You are an academic search assistant."},
            {"role": "user", "content": self.KEYWORD_PROMPT.format(query=user_input)},
        ]
        response = self.llm.response(messages, temperature=0.5)
        try:
            queries = extract(response, "queries")
            return json.loads(queries)
        except:
            return []

    async def _rerank_results(self, user_input: str, results: List[Dict]) -> List[Dict]:
        logging.info("重新排序搜索结果.")

        async def score_paper(result):
            paper_info = f"Title: {result.title}\nAbstract: {result.summary}"
            messages = [
                {"role": "system", "content": "You are an academic relevance scorer."},
                {
                    "role": "user",
                    "content": self.RERANK_PROPMT.format(
                        paper_info=paper_info, input=user_input
                    ),
                },
            ]
            score = float(await self.llm.async_response(messages, temperature=0.3))
            return (score, result)

        # Use asyncio.gather for concurrent async scoring
        tasks = [score_paper(result) for result in results]
        scores = await asyncio.gather(*tasks)
        scores = [score for score in scores if score[0] is not None]

        # Sort results by relevance score and keep top 50%
        num_to_keep = int(len(results) * 0.50)
        scores.sort(key=lambda x: x[0], reverse=True)
        sorted_results = [result for _, result in scores[:num_to_keep]]
        sorted_scores = [score for score, _ in scores[:num_to_keep]]
        return sorted_results, sorted_scores

    async def _search_arxiv(self, user_input: str, max_results: int = 1):
        keywords = self._extract_keywords(user_input)
        logging.info("关键词: %s", keywords)

        # 创建异步任务列表
        tasks = [
            async_get_arxiv_results(
                query=keyword,
                max_results=max_results,
                categories=["cs.AI", "cs.LG"],
                abstract=True,
            )
            for keyword in keywords
        ]
        all_results = await asyncio.gather(*tasks)

        results = set()
        results = []

        for result_set in all_results:
            for result in result_set:
                results.append(result)

        papers, scores = await self._rerank_results(user_input, results)

        return {"papers": papers, "scores": scores}

    def _summarize_paper(self, paper: Dict) -> str:
        title = paper
        abstract = paper.summary

        messages = [
            {
                "role": "system",
                "content": "You are an academic research assistant summarizing papers.",
            },
            {
                "role": "user",
                "content": self.SUMMARY_PAPER_PROMPT.format(
                    paper=title, abstract=abstract
                ),
            },
        ]
        return self.llm.response(messages, temperature=0.7)

    def _summarize_papers(self, papers: List[str], scores: List[str]) -> str:
        logging.info("生成论文总结.")
        messages = [
            {
                "role": "system",
                "content": "You are an academic research assistant summarizing papers.",
            },
            {
                "role": "user",
                "content": self.SUMMARY_PROMPT.format(papers=papers, scores=scores),
            },
        ]
        return self.llm.response(messages, temperature=0.7)


async def main():
    load_dotenv()
    llm = LLM_client("openai/gpt-4o-2024-11-20")
    agent = ResearchAgent(llm)
    user_input = "Find a paper about the application of non-negative matrix decomposition to text clustering or image analysis"
    logging.info("Starting main function with user input: %s", user_input)
    try:
        response = await agent._search_arxiv(user_input)
        print(response)
    except Exception as e:
        logging.error("An error occurred: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
