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

        self.KEYWORD_PROMPT = """您是学术搜索助手，您的任务是生成查询以在arxiv上检索相关论文。如果用户输入中文，请首先将其翻译成英文，并确保翻译的专业性和准确性。用户通常提出一个基础问题，想要进一步了解，但问题的表达形式可能并不理想。

用户输入：{query}

<queries>["query_1", "query_2", ...]</queries>

"""
        self.SUMMARY_PAPER_PROMPT = """请作为学术研究助手对以下论文提供详细总结与分析。

论文标题: {paper}  
论文摘要: {abstract}

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

        self.SUMMARY_PROMPT = """您是一位专业的学术研究助手，请根据提供的相关性评分，对这组论文进行全面的分析和总结。

论文集合: {papers}  
相关性评分：{scores}

请从以下几个方面进行整体分析：

1. **研究主题概览**  
   - 概括这组论文涉及的主要研究领域和核心主题。  
   - 分析研究主题的分布和侧重点。

2. **研究方法与技术路线**  
   - 总结这组论文采用的主要研究方法。  
   - 比较不同论文间的技术路线异同。  
   - 分析方法论的演进趋势。

3. **关键发现与贡献**  
   - 提炼最重要的研究发现和突破。  
   - 评估各项发现的创新性和影响力。  
   - 根据相关性评分分析研究成果的重要程度。

4. **研究脉络与趋势**  
   - 分析论文之间的承接关系和演进路径。  
   - 识别该领域的发展趋势和未来方向。  
   - 指出潜在的研究空白和机会。

请使用简洁专业的语言，确保分析客观全面，突出重点发现。  
总结长度控制在600字以内。

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
