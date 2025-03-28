# PaperCreeper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PaperCreeper 是一个基于人工智能的学术论文智能检索和总结工具，可以帮助研究人员快速获取、筛选和理解最新的科研论文。

## 项目特点

- **智能搜索**：支持自然语言描述查询，自动生成多组专业查询词，多次迭代搜索 arXiv 论文库
- **智能排序**：使用大语言模型对检索结果进行相关性评分和重新排序
- **智能总结**：自动生成论文摘要和关键观点，帮助快速理解论文内容
- **灵活配置**：支持多种大语言模型（OpenAI、Deepseek 等），可根据需求自由切换
- **开源免费**：基于 MIT 许可证开源，可自由使用和修改

## 与官方 arXiv 检索对比的优势

1. **自然语言查询**：支持直接使用自然语言描述研究需求，无需手动构建复杂查询
2. **智能多轮检索**：自动生成多组专业查询词，提高检索覆盖面
3. **智能排序和筛选**：使用大语言模型评估论文与查询的相关性，优先展示最相关的结果
4. **自动总结与见解**：提供论文摘要和关键观点，帮助快速掌握论文内容
5. **批量处理能力**：可同时处理多篇论文，生成综合分析报告

## 安装指南

### 前提条件

- Python 3.8 或更高版本
- 大语言模型 API 访问权限（OpenAI、Deepseek 或其他支持的模型）

### 安装依赖

```bash
git clone https://github.com/askeer25/PaperCreeper.git
cd PaperCreeper
pip install -r requirements.txt
```

### 配置 API 密钥

创建 `.env` 文件并添加您的 API 密钥：

```
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，使用官方 API 时不需要

# Deepseek API 配置（可选）
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=your_deepseek_base_url
```

## 使用方法

### 从 Jupyter Notebook 使用

详见 `main.ipynb` 示例：

```python
from llm_util import *
from arxiv_util import *
from ResearchAgent import *
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置模型
model_name = "gpt-4o"  # 或其他支持的模型
llm = LLM_client(model_name)
agent = ResearchAgent(llm)

# 查询示例
user_input = "搜索近5年内大语言模型推理能力增强相关的论文"
response = search_and_summary(model_name, user_input, num_each_query=5)

# 显示结果
from IPython.display import Markdown, display
display(Markdown(response))
```

### 从命令行使用

运行 Web UI 版本：

```bash
python webui.py
```

然后在浏览器中访问 `http://localhost:7860` 使用图形界面。

## 支持的模型

- **OpenAI 系列**：GPT-3.5-Turbo、GPT-4o 等
- **Deepseek**：deepseek-chat 等
- **其他**：支持兼容 OpenAI API 格式的其他模型服务

## 贡献指南

欢迎贡献代码、报告问题或提出新功能建议！请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 相关项目

- [Semantic Scholar](https://www.semanticscholar.org/)
- [OpenScholar](https://openscholar.allen.ai/)
- [Google Scholar](https://scholar.google.com/)

# Summary

The selected papers present a diverse range of approaches to improving the reasoning capabilities of large language models (LLMs), with applications spanning multi-modal tasks, logical reasoning, and knowledge integration. Below is a detailed analysis of each paper’s research direction, innovations, interconnections, and AI’s potential role in further advancing these areas.

1. **Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent**

   - **Main Research Directions and Contributions:** This paper explores multi-agent strategies to enhance LLM reasoning by integrating Tree of Thoughts (ToT) methods with a Thought Validator agent. The approach aims to improve reasoning reliability by validating and selecting trustworthy reasoning paths.
   - **Innovations and Key Findings:** The novel combination of ToT-based Reasoner agents with a Thought Validator agent allows for more robust and trustworthy reasoning, outperforming existing methods by 5.6% on the GSM8K dataset.
   - **Connections and Relationships:** This work ties into broader efforts to improve LLM reasoning reliability and robustness through structured reasoning path validation.
   - **Relevance to the Broader Field:** The method has implications for AI applications requiring systematic and reliable reasoning, enhancing trust in AI-generated outputs.
   - **AI Insights:** Future improvements could explore more sophisticated validation mechanisms and adaptive reasoning path exploration.
   - **Score:** 10.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2409.11527v2)

2. **Agent Instructs Large Language Models to be General Zero-Shot Reasoners**

   - **Main Research Directions and Contributions:** This paper introduces an autonomous agent to guide LLMs in zero-shot reasoning across various tasks, significantly boosting performance.
   - **Innovations and Key Findings:** The method achieves state-of-the-art zero-shot performance on 20 out of 29 datasets, highlighting a 10.5% improvement over zero-shot chain of thought methods.
   - **Connections and Relationships:** It complements methods aimed at enhancing LLM reasoning flexibility and adaptability in zero-shot settings.
   - **Relevance to the Broader Field:** This research is crucial for expanding the applicability of LLMs in diverse tasks without task-specific training.
   - **AI Insights:** Future research could focus on refining agent instructions and expanding dataset diversity to further enhance generalization.
   - **Score:** 10.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2310.03710v2)

3. **Reasoning in Large Language Models: A Geometric Perspective**

   - **Main Research Directions and Contributions:** This paper examines the reasoning capabilities of LLMs through a geometric lens, linking expressive power to the density of self-attention graphs.
   - **Innovations and Key Findings:** A higher intrinsic dimension of self-attention graphs is correlated with greater expressive capacity, providing a new perspective on LLM reasoning capabilities.
   - **Connections and Relationships:** It offers a theoretical framework that could inform the design of more capable LLMs, aligning with other studies on enhancing LLM reasoning.
   - **Relevance to the Broader Field:** The geometric perspective offers insights into optimizing LLM architectures for improved reasoning.
   - **AI Insights:** Future work could explore practical applications of this geometric framework to guide LLM training and architecture design.
   - **Score:** 9.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2407.02678v1)

4. **Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models**

   - **Main Research Directions and Contributions:** This paper introduces Insight-V, a framework to improve visual reasoning in multi-modal LLMs by generating high-quality reasoning data and utilizing a multi-agent system.
   - **Innovations and Key Findings:** The multi-agent system and iterative DPO algorithm enhance reasoning quality and stability, achieving performance gains on challenging benchmarks.
   - **Connections and Relationships:** This work aligns with efforts to integrate visual and language reasoning, potentially complementing visual-language model advancements.
   - **Relevance to the Broader Field:** Insight-V contributes to the development of more capable multi-modal systems, crucial for complex real-world applications.
   - **AI Insights:** Future research could explore more advanced data generation techniques and agent interactions to further improve multi-modal reasoning.
   - **Score:** 9.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2411.14432v1)

5. **Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models**

   - **Main Research Directions and Contributions:** This paper proposes a graph-constrained reasoning framework to enhance LLM reasoning fidelity by integrating knowledge graph structures.
   - **Innovations and Key Findings:** The KG-Trie index ensures faithful KG-grounded reasoning, achieving state-of-the-art performance on KGQA benchmarks with strong zero-shot generalizability.
   - **Connections and Relationships:** It complements methods focused on integrating structured external knowledge into LLM reasoning processes.
   - **Relevance to the Broader Field:** The approach addresses critical challenges in ensuring reasoning fidelity, particularly in knowledge-intensive tasks.
   - **AI Insights:** Further exploration of hybrid approaches combining structured and unstructured data could yield even more robust reasoning systems.
   - **Score:** 9.0/10
   - **Download:** [arxiv](http://arxiv.org/abs/2410.13080v1)

6. **Towards LogiGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models**

   - **Main Research Directions and Contributions:** This work reviews logical reasoning in LLMs and introduces the LogiGLUE benchmark to assess LLM performance across various logical reasoning tasks.
   - **Innovations and Key Findings:** LogiGLUE provides a comprehensive benchmark for evaluating and improving LLM logical reasoning, highlighting disparities across reasoning types.
   - **Connections and Relationships:** It offers benchmarking resources that complement studies focused on enhancing reasoning capabilities in LLMs.
   - **Relevance to the Broader Field:** Understanding and improving LLM logical reasoning is crucial for applications requiring precise and reliable decision-making.
   - **AI Insights:** Further development of task-specific fine-tuning and evaluation methods could improve reasoning performance across diverse logical challenges.
   - **Score:** 9.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2310.00836v3)

7. **Improving Arithmetic Reasoning Ability of Large Language Models through Relation Tuples, Verification and Dynamic Feedback**

   - **Main Research Directions and Contributions:** This paper proposes using relation tuples in reasoning steps to improve the arithmetic reasoning capabilities of LLMs.
   - **Innovations and Key Findings:** By combining verification processes and dynamic feedback, the approach enhances the readability and verifiability of reasoning steps.
   - **Connections and Relationships:** It aligns with broader efforts to improve specific reasoning tasks like arithmetic, which are foundational to more complex reasoning challenges.
   - **Relevance to the Broader Field:** Enhancing arithmetic reasoning is critical for LLM applications in STEM fields and quantitative analysis.
   - **AI Insights:** Incorporating more sophisticated verification and feedback mechanisms could further increase reasoning accuracy and reliability.
   - **Score:** 9.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2406.17873v1)

8. **MISR: Measuring Instrumental Self-Reasoning in Frontier Models**

   - **Main Research Directions and Contributions:** This paper introduces tasks to evaluate the instrumental self-reasoning abilities of LLM agents, focusing on agentic tasks like self-modification and knowledge seeking.
   - **Innovations and Key Findings:** The evaluations reveal that self-reasoning emerges in advanced models, providing a metric for assessing future improvements in model reasoning capabilities.
   - **Connections and Relationships:** It provides a framework for evaluating self-reasoning, complementing studies on reasoning capabilities and agentic behaviors in LLMs.
   - **Relevance to the Broader Field:** A deeper understanding of self-reasoning is vital for developing autonomous AI systems capable of complex decision-making.
   - **AI Insights:** Future work could refine evaluation metrics and explore methods to enhance self-reasoning abilities in LLMs.
   - **Score:** 9.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2412.03904v1)

9. **Logic-Enhanced Language Model Agents for Trustworthy Social Simulations**

   - **Main Research Directions and Contributions:** This paper presents the LELMA framework to improve the trustworthiness of social simulations by integrating LLMs with symbolic AI for logical verification.
   - **Innovations and Key Findings:** The integration of logical verification enhances reasoning accuracy in game-theoretic scenarios, highlighting limitations in current LLMs.
   - **Connections and Relationships:** It aligns with efforts to improve AI reasoning through symbolic and logical methods, enhancing trust in AI-generated social simulations.
   - **Relevance to the Broader Field:** The framework is crucial for applications in AI-driven social simulations and decision-making systems.
   - **AI Insights:** Further integration of symbolic methods could enhance reasoning accuracy and reduce hallucinations in complex scenarios.
   - **Score:** 9.0/10
   - **Download:** [arXiv](http://arxiv.org/abs/2408.16081v1)

10. **Laying the Foundation First? Investigating the Generalization from Atomic Skills to Complex Reasoning Tasks**

    - **Main Research Directions and Contributions:** This paper explores the generalization of atomic skills to complex reasoning tasks, proposing a hierarchical curriculum learning strategy.
    - **Innovations and Key Findings:** The strategy successfully induces skill generalization, improving LLM performance on complex reasoning tasks and highlighting the interdependence of atomic and complex skills.
    - **Connections and Relationships:** It complements studies focused on skill generalization and curriculum learning in enhancing LLM capabilities.
    - **Relevance to the Broader Field:** Understanding skill generalization is vital for developing LLMs capable of tackling complex, multifaceted tasks.
    - **AI Insights:** Future work could explore adaptive learning strategies to further enhance skill generalization and reasoning performance.
    - **Score:** 9.0/10
    - **Download:** [arXiv](http://arxiv.org/abs/2403.09479v1)

11. **Enhance Reasoning Ability of Visual-Language Models via Large Language Models**

    - **Main Research Directions and Contributions:** This paper proposes the TReE method to transfer reasoning abilities from LLMs to visual language models (VLMs) in zero-shot scenarios.
    - **Innovations and Key Findings:** The three-stage process (observation, thinking, re-thinking) enhances VLM reasoning, demonstrating significant improvements in zero-shot settings.
    - **Connections and Relationships:** It aligns with efforts to integrate reasoning capabilities across modalities, enhancing the functionality of VLMs.
    - **Relevance to the Broader Field:** The approach is significant for applications requiring advanced visual-language reasoning, such as image captioning and understanding.
    - **AI Insights:** Future research could explore more seamless integration of reasoning across modalities to further improve VLM capabilities.
    - **Score:** 8.0/10
    - **Download:** [arXiv](http://arxiv.org/abs/2305.13267v1)

12. **Gemini in Reasoning: Unveiling Commonsense in Multimodal Large Language Models**

    - **Main Research Directions and Contributions:** This study evaluates the commonsense reasoning capabilities of the Gemini MLLM across various datasets, highlighting its competitive performance.
    - **Innovations and Key Findings:** Despite initial benchmarks, comprehensive evaluation reveals Gemini's strong commonsense reasoning, particularly in multi-modal tasks.
    - **Connections and Relationships:** It provides insights into the commonsense reasoning capabilities of MLLMs, complementing studies focused on enhancing multi-modal model performance.
    - **Relevance to the Broader Field:** Understanding and improving commonsense reasoning in MLLMs is crucial for applications requiring nuanced decision-making and understanding.
    - **AI Insights:** Future advancements could focus on enhancing commonsense integration across modalities for more holistic reasoning capabilities.
    - **Score:** 8.0/10
    - **Download:** [arXiv](http://arxiv.org/abs/2312.17661v1)

Overall, these papers contribute significantly to the ongoing development of reasoning capabilities in LLMs, addressing challenges related to trustworthiness, generalization, and multi-modal integration. The collective insights and innovations provide a strong foundation for future advancements in AI reasoning systems.


## 与arXiv官方网站搜索功能对比的优势

1. PaperCreeper支持自然语句描述的查询，并根据查询语句自动生成若干查询词，多次迭代搜索，优化搜索结果。
2. PaperCreeper支持对检索结果进行打分和重新排序，提升检索质量。
3. PaperCreeper支持LLM对检索结果的阅读和总结，并生成独特见解，方便用户预览文章内容，快速聚焦有用的论文。


### 相关工具
1. [Semantic Scholar](https://www.semanticscholar.org/)
2. [OpenScholar](https://openscholar.allen.ai/)
3. [Google Scholar](https://scholar.google.com/)
