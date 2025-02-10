import json
from typing import List, Dict
from arxiv_util import (
    get_arxiv_results,
    get_arxiv_message,
)  # Ensure these functions are defined
from dataclasses import dataclass
from llm_util import *
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class SearchCriteria:
    keywords: List[str]
    field: str = None
    published_year: int = None
    max_results: int = 5


class ResearchAgent:
    def __init__(self, llm_client: LLM_client):
        self.search_history = []
        self.feedback_history = []
        self.llm = llm_client

        # Define prompt templates
        self.KEYWORD_PROMPT = """You are an academic search assistant, your task is to generate queries to retrieve relevant papers on arxiv.
If the user enters Chinese, please translate it into English first and ensure professionalism and accuracy.
User input: {query}

Output strictly in the following format:
<queries>["query1", "query2", ...]</queries>
"""
        self.FIELD_PROMPT = """You are an academic search assistant, your task is to determine the main arXiv domain for the following query.

Available fields: cs, physics, math, econ, etc.

User input: {query}

Output strictly in the following format:
<field>'field_name'</field>
"""
        self.published_year_PROMPT = """You are an academic search assistant, your task is to extract the publication date range from the following query. If no specific dates are mentioned, return None.

Returns a year number (e.g., "2020") and relative time periods (e.g., "last 5 years").

User input: {query}

Output strictly in the following format:
<published_year>'datetime'</published_year>
        """
        self.SUMMARY_PROMPT = """You are an academic research assistant. Please provide a detailed summary and analysis of the following collection of papers, taking into account the given relevance scores. 

Consider the papers and their scores: {papers}

Your summary should address the following aspects:

1. Main Research Directions and Contributions: Describe the core research areas, objectives, and main contributions of each paper. What problem are the authors trying to solve, and how do they approach it?

2. Innovations and Key Findings: Highlight the novel methods, techniques, or findings that distinguish each paper. What is unique about their approach or results?

3. Connections and Relationships Between Papers: Identify any links between the papers, whether they address similar problems, use complementary methods, or build on each other’s findings. Are there overarching trends or common themes across the papers?

4. Relevance to the Broader Field: Briefly discuss the implications of these papers in their respective fields and, where applicable, how AI techniques or tools have been integrated into their approaches.

5. Additional AI Insights: Provide a concise perspective on how AI and machine learning methods could influence or improve upon the approaches discussed in the papers, based on the emerging trends and challenges identified.

Don't leave out any papers.

Formatting Requirements:

- Scores: Incorporate the relevance score (from 1 to 10) at the end of each paper's summary to highlight its importance to the overall discussion.

- ArXiv URL: Provide the direct link to the arXiv page of each paper.


Example Structure for Summaries:

**Example**

The selected papers span a range of topics, including multimodal models, quantum computing, machine learning, and computational chemistry. Below is a detailed analysis of each paper’s research direction, innovations, interconnections, and AI’s potential role in further advancing these areas.

1. *VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction*

Main Research Direction and Contributions: This paper focuses on integrating real-time vision and speech interaction into multimodal language models. The authors propose a multi-stage training methodology that enables seamless communication between visual and auditory inputs without requiring separate ASR (automatic speech recognition) or TTS (text-to-speech) modules.
Innovations and Key Findings: The main innovation is the end-to-end model that handles both vision and speech tasks in real-time. This allows for more fluid interaction, bypassing traditional modular approaches.
Connections and Relationships: This work is closely related to advancements in multimodal models and speech recognition technologies, potentially aligning with other papers focused on integrating visual, auditory, and textual data in unified systems.
Relevance to the Broader Field: The paper sets a new benchmark for real-time interaction in multimodal systems, with wide applications in robotics, virtual assistants, and autonomous systems.
AI Insights: AI-driven techniques such as unsupervised learning and reinforcement learning could further enhance the adaptability and performance of such systems in dynamic real-world environments.
Score: 9/10
Download: [arXiv](http://arxiv.org/abs/2501.01957v1)

2. *Metadata Conditioning Accelerates Language Model Pre-training*

Main Research Direction and Contributions: This paper addresses language model pre-training by incorporating external metadata (e.g., URLs) during the training process. The proposed MeCo method helps accelerate model training, requiring fewer data while maintaining high performance.
Innovations and Key Findings: The integration of metadata allows for more efficient use of resources, with the method producing more steerable and customizable language models.
Connections and Relationships: The research connects with ongoing efforts to improve the efficiency and scalability of language model training, particularly in terms of data utilization and model fine-tuning.
Relevance to the Broader Field: This approach can significantly impact areas such as NLP, where rapid model adaptation and efficient resource use are critical.
AI Insights: Future AI advancements could explore ways to leverage more diverse metadata types, such as semantic context, for even more robust and domain-specific language models.
Score: 8/10
Download: [arXiv](http://arxiv.org/abs/2501.01956v1)

[... and so on for all papers.]
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

    def _extract_field(self, text: str) -> str:
        logging.info("从用户输入中识别研究领域.")
        messages = [
            {"role": "system", "content": "You are an academic search assistant."},
            {"role": "user", "content": self.FIELD_PROMPT.format(query=text)},
        ]
        response = self.llm.response(messages, temperature=0.5)
        try:
            field = extract(response, "field")
            return field
        except:
            return ""

    def _extract_date(self, text: str) -> str:
        logging.info("从用户输入中提取日期.")
        messages = [
            {"role": "system", "content": "You are a date extraction assistant."},
            {"role": "user", "content": self.published_year_PROMPT.format(query=text)},
        ]
        response = self.llm.response(messages, temperature=0.3)
        try:
            published_year = extract(response, "published_year")
            return int(published_year)
        except:
            return None

    def del_earlier_results(
        self, published_year: int, results: List[Dict]
    ) -> List[Dict]:

        logging.info("删除早期搜索结果.")
        if published_year is None:
            return results
        else:
            return [
                result for result in results if result.published.year >= published_year
            ]

    def _rerank_results(self, text: str, results: List[Dict]) -> List[Dict]:
        logging.info("重新排序搜索结果.")

        scores = []
        for result in results:
            paper_info = f"Title: {result.title}\nAbstract: {result.summary}"
            messages = [
                {"role": "system", "content": "You are an academic relevance scorer."},
                {
                    "role": "user",
                    "content": self.RERANK_PROPMT.format(
                        paper_info=paper_info, input=text
                    ),
                },
            ]
            score = float(self.llm.response(messages, temperature=0.3))
            scores.append((score, result))

        # Sort results by relevance score and keep top 50%
        num_to_keep = int(len(results) * 0.50)
        scores.sort(key=lambda x: x[0], reverse=True)
        sorted_results = [result for _, result in scores[:num_to_keep]]
        sorted_scores = [score for score, _ in scores[:num_to_keep]]
        return sorted_results, sorted_scores

    def _generate_summary(self, papers: List[str], scores: List[float]) -> str:
        logging.info("生成论文总结.")
        papers_with_scores = [
            f"{paper}\nRelevance Score: {score}/10"
            for paper, score in zip(papers, scores)
        ]

        messages = [
            {
                "role": "system",
                "content": "You are an academic research assistant summarizing papers.",
            },
            {
                "role": "user",
                "content": self.SUMMARY_PROMPT.format(papers=papers_with_scores),
            },
        ]
        return self.llm.response(messages, temperature=0.7)

    #     def reflect(self, feedback: str = None):
    #         logging.info("反思改进搜索策略.")
    #         if not feedback:
    #             return
    #         self.feedback_history.append(feedback)

    #         reflection_prompt = f"""You are an academic search assistant, optimize your search strategy based on the following information:

    # Search history:
    # {json.dumps(self.search_history[-1], indent=2)}

    # User feedback:
    # {feedback}

    # Please analyze the existing problems and make suggestions for improvement.
    # Output JSON format:
    # {{
    # "analysis": "Analysis result ",
    # "improvements": [" Improvements 1", "Improvements 2",...]
    # }}"""
    #         messages = [
    #             {
    #                 "role": "system",
    #                 "content": "You are an academic search assistant who is good at summarizing and reflecting.",
    #             },
    #             {"role": "user", "content": reflection_prompt},
    #         ]

    #         response = self.llm.response(messages, temperature=0.7)
    #         try:
    #             improvements = json.loads(response)
    #             return improvements
    #         except:
    #             return None

    def analyze_query(self, user_input: str) -> SearchCriteria:
        logging.info("分析用户查询.")
        keywords = self._extract_keywords(user_input)
        published_year = self._extract_date(user_input)
        field = self._extract_field(user_input)
        return SearchCriteria(
            keywords=keywords, published_year=published_year, field=field
        )

    # def optimize_query(self, criteria: SearchCriteria) -> str:
    #     logging.info("优化查询.")
    #     query_parts = []
    #     keyword_query = " AND ".join(f'"{k}"' for k in criteria.keywords)
    #     query_parts.append(keyword_query)
    #     if criteria.start_date and criteria.end_date:
    #         query_parts.append(
    #             f"submittedDate:[{criteria.start_date} TO {criteria.end_date}]"
    #         )
    #     if criteria.field:
    #         query_parts.append(f"cat:{criteria.field}")
    #     return " AND ".join(query_parts)

    def search_and_summarize(
        self, user_input: str, num_each_query: int = 5, feedback: str = None
    ) -> str:
        # 主要函数
        logging.info("开始搜索和总结.")
        criteria = self.analyze_query(user_input)
        criteria.max_results = num_each_query

        # 优化查询
        # query = self.optimize_query(criteria)

        logging.info(f"提取查询关键词: {criteria.keywords}")

        results = []
        for keyword in criteria.keywords:
            query = keyword
            result = get_arxiv_results(query, max_results=criteria.max_results)
            results.extend(result)

        # # 删除早期结果
        # results = self.del_earlier_results(criteria.published_year, results)

        # 重新排序结果
        results, scores = self._rerank_results(user_input, results)

        # print(results)

        messages = []
        for result in results:
            message = get_arxiv_message(result)
            messages.append(message)

        # 生成总结
        summary = self._generate_summary(messages, scores)

        logging.info("Search and summary completed.")
        return summary


def main():
    load_dotenv()
    llm = LLM_client("openai/gpt-4o-2024-11-20")
    agent = ResearchAgent(llm)
    user_input = "Find a paper about the application of non-negative matrix decomposition to text clustering or image analysis"
    logging.info("Starting main function with user input: %s", user_input)
    try:
        response = agent.search_and_summarize(user_input, num_each_query=3)
        print(response)
    except Exception as e:
        logging.error("An error occurred: %s", e)


if __name__ == "__main__":
    main()
