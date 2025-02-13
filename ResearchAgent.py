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

        self.KEYWORD_PROMPT = """You are an academic search assistant tasked with generating queries to search for relevant papers on arXiv.
If the user inputs in Chinese, first translate it into English and ensure the translation is professional and accurate.
The user typically poses a basic question wanting to know more, but the question may not always be clearly articulated.

Given user's input: {query}

Generate a list of search queries:
<queries>["query_1", "query_2", ...]</queries>

"""
        self.SUMMARY_PAPER_PROMPT = """请提供一份详细的论文总结与分析, 使用中文回答：

**论文标题**: {paper}  
**论文摘要**: {abstract}

您的总结应包括以下几个方面：

1. **主要研究方向与贡献**
   - 描述论文的核心研究领域、目标及主要贡献。
   - 作者试图解决什么问题？
   - 采用了哪些方法来解决问题？

2. **创新点与关键发现**
   - 突出论文中的新颖方法、技术或研究成果。
   - 这些新内容如何使论文区别于其他相关研究？
   - 其方法或结果有何独特之处？

请使用**Markdown**语法作答，仅使用列表、文本、加粗字体，避免使用其他格式。 
请确保您的回答专业准确，字数不超过300字。

"""

        self.SUMMARY_PROMPT = """作为专业的学术研究助理，您的任务是根据提供的相关分数对一组论文进行全面分析和总结。

您需要在分析中涉及以下方面：

1. 研究主题概述
   - 总结一组{papers}涵盖的主要研究领域和核心主题。
   - 分析研究主题的分布和重点。

2. 研究方法和技术手段
   - 总结{papers}中采用的主要研究方法。
   - 比较使用的技术方法。
   - 分析研究方法的发展趋势。

3. 主要发现和贡献
   - 提取最重要的研究发现和突破。
   - 评估每一项发现的创新性和影响力。

4. 研究连续性和趋势
   - 分析{papers}之间的相互关联和演变路径。
   - 确定该领域的发展趋势和未来方向。
   - 突出潜在的研究空白和机遇。

在论文中提到某一篇论文时，请使用标准的IEEE引用格式。

您的分析应简洁明了，使用专业术语以确保客观和全面，同时强调重点发现。摘要应限制在600字以内。

确保您的分析提供对研究论文的全面理解，详细涵盖指定的各个方面。

    """

        self.RERANK_PROPMT = """你是一个学术相关性评分员，根据用户兴趣对论文的相关性进行评分，评分范围为0-10。

用户输入： {input}

请考虑以下论文： {paper_info}

请根据以下标准给出该论文的相关性评分（0-10）： 0 = 完全不相关 5 = 中等相关 10 = 极其相关

请仅提供一个0-10之间的数字评分，不要附加其他文字。

"""

    def _extract_keywords(self, user_input: str) -> List[str]:
        logging.info("从用户输入中提取关键词.")
        messages = [
            {"role": "system", "content": "You are an academic search assistant."},
            {"role": "user", "content": self.KEYWORD_PROMPT.format(query=user_input)},
        ]
        response = self.llm.response(messages, temperature=0.4)
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
            score = float(await self.llm.async_response(messages, temperature=0.4))
            return (score, result)

        tasks = [score_paper(result) for result in results]
        scores = await asyncio.gather(*tasks)
        scores = [score for score in scores if score[0] is not None]

        num_to_keep = int(len(results) * 0.50)
        scores.sort(key=lambda x: x[0], reverse=True)
        sorted_results = [result for _, result in scores[:num_to_keep]]
        sorted_scores = [score for score, _ in scores[:num_to_keep]]
        unique_titles = set()
        deduped_results = []
        deduped_scores = []
        for score, result in zip(sorted_scores, sorted_results):
            if result.title not in unique_titles:
                unique_titles.add(result.title)
                deduped_results.append(result)
                deduped_scores.append(score)
        sorted_results = deduped_results
        sorted_scores = deduped_scores
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
        return self.llm.response(messages, temperature=0.4)

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
        return self.llm.response(messages, temperature=0.4)


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
