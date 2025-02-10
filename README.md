# Paper Creeper
## 介绍
这是一个简单的论文辅助检索工具，可以从arXiv智能检索科学论文。

## 安装

### 获取api_key
支持OpenAI、Deepseek以及其它代理商(例如ChatAnyWhere)

### 安装依赖项


```python
! pip install arxiv openai python-dotenv
```

或者


```python
! pip install -r requirements.txt
```

## 运行

### 外部函数


```python
from llm_util import *
from arxiv_util import *
from ResearchAgent import *
```

### 主函数
参数说明：
- `model_name`: LLM模型名称
- `user_input`: 文献检索查询语句


```python
def search_and_summary(model_name: str="gpt-4o", user_input:str=None, num_each_query:int=5):
    load_dotenv()
    llm = LLM_client(model_name)
    agent = ResearchAgent(llm)
    logging.info("开始主函数, 用户输入: %s", user_input)
    try:
        response = agent.search_and_summarize(user_input, num_each_query)
        # print(response)
        return response
    except Exception as e:
        logging.error("An error occurred: %s", e)
        return None
```

### 设置模型名称
默认支持的模型名称有：
- deepseek-chat: 需要在`.env`环境中指定`DEEPSEEK_API_KEY`和`DEEPSEEK_BASE_URL`
- GPT*: 需要在`.env`环境中指定`OPENAI_API_KEY`和`OPENAI_BASE_URL`


```python
# 设置
model_name = "gpt-4o"
```


```python
# display util
from IPython.display import Markdown, display

def display_response_as_markdown(response):
    display(Markdown(response))

```

### 检索论文


```python
# 检索并显示总结结果
response = search_and_summary(model_name,
                              user_input="""Search the papers about the enhancement of reason ability of large language model agents in recent 5 years.""",
                              num_each_query=3)
display_response_as_markdown(response)

```

    2025-01-07 09:24:20,715 - INFO - Starting main function with user input: Search the papers about the enhancement of reason ability of large language model agents in recent 5 years.
    2025-01-07 09:24:20,716 - INFO - 开始搜索和总结.
    2025-01-07 09:24:20,716 - INFO - 分析用户查询.
    2025-01-07 09:24:20,717 - INFO - 从用户输入中提取关键词.
    2025-01-07 09:24:23,183 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:24:23,186 - INFO - 从用户输入中提取日期.
    2025-01-07 09:24:23,912 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:24:23,914 - INFO - 从用户输入中识别研究领域.
    2025-01-07 09:24:24,598 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:24:24,600 - INFO - 提取查询关键词: ['enhancement of reasoning ability in large language models', 'improving reasoning in large language model agents', 'advancements in reasoning capabilities of large language models', 'reasoning enhancement in large language models', 'recent developments in reasoning for large language models', 'large language models reasoning ability improvement', 'enhanced reasoning in AI language models', 'reasoning skills enhancement in large language models', 'large language models reasoning capabilities 2018-2023', 'reasoning ability advancements in language models']
    2025-01-07 09:24:24,601 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=enhancement+of+reasoning+ability+in+large+language+models&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D9BC040>
    

    2025-01-07 09:24:29,754 - INFO - Got first page: 100 of 2634498 total results
    2025-01-07 09:24:29,756 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=improving+reasoning+in+large+language+model+agents&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D9BBEC0>
    

    2025-01-07 09:24:32,422 - INFO - Got first page: 100 of 2520750 total results
    2025-01-07 09:24:32,423 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=advancements+in+reasoning+capabilities+of+large+language+models&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D6836F0>
    

    2025-01-07 09:24:35,105 - INFO - Got first page: 100 of 2634482 total results
    2025-01-07 09:24:35,107 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=reasoning+enhancement+in+large+language+models&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D80E020>
    

    2025-01-07 09:24:38,182 - INFO - Got first page: 100 of 2515594 total results
    2025-01-07 09:24:38,183 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=recent+developments+in+reasoning+for+large+language+models&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9C958180>
    

    2025-01-07 09:24:40,837 - INFO - Got first page: 100 of 2599191 total results
    2025-01-07 09:24:40,839 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=large+language+models+reasoning+ability+improvement&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D80E020>
    

    2025-01-07 09:24:43,173 - INFO - Got first page: 100 of 1367381 total results
    2025-01-07 09:24:43,174 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=enhanced+reasoning+in+AI+language+models&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D99A570>
    

    2025-01-07 09:24:46,203 - INFO - Got first page: 100 of 2507171 total results
    2025-01-07 09:24:46,204 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=reasoning+skills+enhancement+in+large+language+models&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D8A0950>
    

    2025-01-07 09:24:47,935 - INFO - Got first page: 10 of 2515667 total results
    2025-01-07 09:24:47,936 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=large+language+models+reasoning+capabilities+2018-2023&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9D8A18F0>
    

    2025-01-07 09:24:50,232 - INFO - Got first page: 100 of 1261513 total results
    2025-01-07 09:24:50,233 - INFO - Requesting page (first: True, try: 0): https://export.arxiv.org/api/query?search_query=reasoning+ability+advancements+in+language+models&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100
    

    <itertools.islice object at 0x000001FC9C958180>
    

    2025-01-07 09:24:52,917 - INFO - Got first page: 100 of 2506313 total results
    2025-01-07 09:24:52,918 - INFO - 删除早期搜索结果.
    2025-01-07 09:24:52,919 - INFO - 重新排序搜索结果.
    2025-01-07 09:24:53,627 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:24:57,162 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:24:59,032 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:24:59,647 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:00,400 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:02,247 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:02,906 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:03,520 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:04,311 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:05,133 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:05,748 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:06,306 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:06,929 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:07,519 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:08,140 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:08,709 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:09,642 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:10,193 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:12,728 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:13,358 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:13,947 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:14,562 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:15,116 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:15,713 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:16,272 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:17,071 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:17,625 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:18,267 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:18,997 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:19,557 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:19,560 - INFO - 生成论文总结.
    

    **Title:** Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent
    **Authors:** Fatemeh Haji, Mazal Bethany, Maryam Tabar, Jason Chiang, Anthony Rios, Peyman Najafirad
    **Published:** 2024
    **Summary:** Multi-agent strategies have emerged as a promising approach to enhance the reasoning abilities of Large Language Models (LLMs) by assigning specialized roles in the problem-solving process. Concurrently, Tree of Thoughts (ToT) methods have shown potential in improving reasoning for complex question-answering tasks by exploring diverse reasoning paths. A critical limitation in multi-agent reasoning is the 'Reasoner' agent's shallow exploration of reasoning paths. While ToT strategies could help mitigate this problem, they may generate flawed reasoning branches, which could harm the trustworthiness of the final answer. To leverage the strengths of both multi-agent reasoning and ToT strategies, we introduce a novel approach combining ToT-based Reasoner agents with a Thought Validator agent. Multiple Reasoner agents operate in parallel, employing ToT to explore diverse reasoning paths. The Thought Validator then scrutinizes these paths, considering a Reasoner's conclusion only if its reasoning is valid. This method enables a more robust voting strategy by discarding faulty reasoning paths, enhancing the system's ability to tackle tasks requiring systematic and trustworthy reasoning. Our method demonstrates superior performance compared to existing techniques when evaluated on the GSM8K dataset, outperforming the standard ToT strategy by an average 5.6% across four LLMs. The code and related content can be found in: https://github.com/SecureAIAutonomyLab/MA-ToT
    **URL:** http://arxiv.org/abs/2409.11527v2
    **Title:** Agent Instructs Large Language Models to be General Zero-Shot Reasoners
    **Authors:** Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, Chenguang Wang
    **Published:** 2023
    **Summary:** We introduce a method to improve the zero-shot reasoning abilities of large language models on general language understanding tasks. Specifically, we build an autonomous agent to instruct the reasoning process of large language models. We show this approach further unleashes the zero-shot reasoning abilities of large language models to more tasks. We study the performance of our method on a wide set of datasets spanning generation, classification, and reasoning. We show that our method generalizes to most tasks and obtains state-of-the-art zero-shot performance on 20 of the 29 datasets that we evaluate. For instance, our method boosts the performance of state-of-the-art large language models by a large margin, including Vicuna-13b (13.3%), Llama-2-70b-chat (23.2%), and GPT-3.5 Turbo (17.0%). Compared to zero-shot chain of thought, our improvement in reasoning is striking, with an average increase of 10.5%. With our method, Llama-2-70b-chat outperforms zero-shot GPT-3.5 Turbo by 10.2%.
    **URL:** http://arxiv.org/abs/2310.03710v2
    **Title:** Reasoning in Large Language Models: A Geometric Perspective
    **Authors:** Romain Cosentino, Sarath Shekkizhar
    **Published:** 2024
    **Summary:** The advancement of large language models (LLMs) for real-world applications hinges critically on enhancing their reasoning capabilities. In this work, we explore the reasoning abilities of large language models (LLMs) through their geometrical understanding. We establish a connection between the expressive power of LLMs and the density of their self-attention graphs. Our analysis demonstrates that the density of these graphs defines the intrinsic dimension of the inputs to the MLP blocks. We demonstrate through theoretical analysis and toy examples that a higher intrinsic dimension implies a greater expressive capacity of the LLM. We further provide empirical evidence linking this geometric framework to recent advancements in methods aimed at enhancing the reasoning capabilities of LLMs.
    **URL:** http://arxiv.org/abs/2407.02678v1
    **Title:** Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models
    **Authors:** Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, Ziwei Liu
    **Published:** 2024
    **Summary:** Large Language Models (LLMs) demonstrate enhanced capabilities and reliability by reasoning more, evolving from Chain-of-Thought prompting to product-level solutions like OpenAI o1. Despite various efforts to improve LLM reasoning, high-quality long-chain reasoning data and optimized training pipelines still remain inadequately explored in vision-language tasks. In this paper, we present Insight-V, an early effort to 1) scalably produce long and robust reasoning data for complex multi-modal tasks, and 2) an effective training pipeline to enhance the reasoning capabilities of multi-modal large language models (MLLMs). Specifically, to create long and structured reasoning data without human labor, we design a two-step pipeline with a progressive strategy to generate sufficiently long and diverse reasoning paths and a multi-granularity assessment method to ensure data quality. We observe that directly supervising MLLMs with such long and complex reasoning data will not yield ideal reasoning ability. To tackle this problem, we design a multi-agent system consisting of a reasoning agent dedicated to performing long-chain reasoning and a summary agent trained to judge and summarize reasoning results. We further incorporate an iterative DPO algorithm to enhance the reasoning agent's generation stability and quality. Based on the popular LLaVA-NeXT model and our stronger base MLLM, we demonstrate significant performance gains across challenging multi-modal benchmarks requiring visual reasoning. Benefiting from our multi-agent system, Insight-V can also easily maintain or improve performance on perception-focused multi-modal tasks.
    **URL:** http://arxiv.org/abs/2411.14432v1
    **Title:** Reasoning in Large Language Models: A Geometric Perspective
    **Authors:** Romain Cosentino, Sarath Shekkizhar
    **Published:** 2024
    **Summary:** The advancement of large language models (LLMs) for real-world applications hinges critically on enhancing their reasoning capabilities. In this work, we explore the reasoning abilities of large language models (LLMs) through their geometrical understanding. We establish a connection between the expressive power of LLMs and the density of their self-attention graphs. Our analysis demonstrates that the density of these graphs defines the intrinsic dimension of the inputs to the MLP blocks. We demonstrate through theoretical analysis and toy examples that a higher intrinsic dimension implies a greater expressive capacity of the LLM. We further provide empirical evidence linking this geometric framework to recent advancements in methods aimed at enhancing the reasoning capabilities of LLMs.
    **URL:** http://arxiv.org/abs/2407.02678v1
    **Title:** Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models
    **Authors:** Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza Haffari, Shirui Pan
    **Published:** 2024
    **Summary:** Large language models (LLMs) have demonstrated impressive reasoning abilities, but they still struggle with faithful reasoning due to knowledge gaps and hallucinations. To address these issues, knowledge graphs (KGs) have been utilized to enhance LLM reasoning through their structured knowledge. However, existing KG-enhanced methods, either retrieval-based or agent-based, encounter difficulties in accurately retrieving knowledge and efficiently traversing KGs at scale. In this work, we introduce graph-constrained reasoning (GCR), a novel framework that bridges structured knowledge in KGs with unstructured reasoning in LLMs. To eliminate hallucinations, GCR ensures faithful KG-grounded reasoning by integrating KG structure into the LLM decoding process through KG-Trie, a trie-based index that encodes KG reasoning paths. KG-Trie constrains the decoding process, allowing LLMs to directly reason on graphs and generate faithful reasoning paths grounded in KGs. Additionally, GCR leverages a lightweight KG-specialized LLM for graph-constrained reasoning alongside a powerful general LLM for inductive reasoning over multiple reasoning paths, resulting in accurate reasoning with zero reasoning hallucination. Extensive experiments on several KGQA benchmarks demonstrate that GCR achieves state-of-the-art performance and exhibits strong zero-shot generalizability to unseen KGs without additional training.
    **URL:** http://arxiv.org/abs/2410.13080v1
    **Title:** Towards LogiGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models
    **Authors:** Man Luo, Shrinidhi Kumbhar, Ming shen, Mihir Parmar, Neeraj Varshney, Pratyay Banerjee, Somak Aditya, Chitta Baral
    **Published:** 2023
    **Summary:** Logical reasoning is fundamental for humans yet presents a substantial challenge in the domain of Artificial Intelligence. Initially, researchers used Knowledge Representation and Reasoning (KR) systems that did not scale and required non-trivial manual effort. Recently, the emergence of large language models (LLMs) has demonstrated the ability to overcome various limitations of formal Knowledge Representation (KR) systems. Consequently, there's a growing interest in using LLMs for logical reasoning via natural language. This work strives to understand the proficiency of LLMs in logical reasoning by offering a brief review of the latest progress in this area; with a focus on the logical reasoning datasets, tasks, and the methods adopted to utilize LLMs for reasoning. To offer a thorough analysis, we have compiled a benchmark titled LogiGLUE. This includes 24 varied datasets encompassing deductive, abductive, and inductive reasoning. Utilizing LogiGLUE as a foundation, we have trained an instruction fine-tuned language model, resulting in LogiT5. We study single-task training, multi-task training, and "chain-of-thought" knowledge distillation fine-tuning technique to assess the performance of model across the different logical reasoning categories. We also assess various LLMs using LogiGLUE, and the findings indicate that LLMs excel most in abductive reasoning, followed by deductive reasoning, while they are least effective at inductive reasoning. We aim to shed light on the capabilities and potential pathways for enhancing logical reasoning proficiency in LLMs, paving the way for more advanced and nuanced developments in this critical field.
    **URL:** http://arxiv.org/abs/2310.00836v3
    **Title:** Improving Arithmetic Reasoning Ability of Large Language Models through Relation Tuples, Verification and Dynamic Feedback
    **Authors:** Zhongtao Miao, Kaiyan Zhao, Yoshimasa Tsuruoka
    **Published:** 2024
    **Summary:** Current representations used in reasoning steps of large language models can mostly be categorized into two main types: (1) natural language, which is difficult to verify; and (2) non-natural language, usually programming code, which is difficult for people who are unfamiliar with coding to read. In this paper, we propose to use a semi-structured form to represent reasoning steps of large language models. Specifically, we use relation tuples, which are not only human-readable but also machine-friendly and easier to verify than natural language. We implement a framework that includes three main components: (1) introducing relation tuples into the reasoning steps of large language models; (2) implementing an automatic verification process of reasoning steps with a local code interpreter based on relation tuples; and (3) integrating a simple and effective dynamic feedback mechanism, which we found helpful for self-improvement of large language models. The experimental results on various arithmetic datasets demonstrate the effectiveness of our method in improving the arithmetic reasoning ability of large language models. The source code is available at https://github.com/gpgg/art.
    **URL:** http://arxiv.org/abs/2406.17873v1
    **Title:** MISR: Measuring Instrumental Self-Reasoning in Frontier Models
    **Authors:** Kai Fronsdal, David Lindner
    **Published:** 2024
    **Summary:** We propose a suite of tasks to evaluate the instrumental self-reasoning ability of large language model (LLM) agents. Instrumental self-reasoning ability could improve adaptability and enable self-modification, but it could also pose significant risks, such as enabling deceptive alignment. Prior work has only evaluated self-reasoning in non-agentic settings or in limited domains. In this paper, we propose evaluations for instrumental self-reasoning ability in agentic tasks in a wide range of scenarios, including self-modification, knowledge seeking, and opaque self-reasoning. We evaluate agents built using state-of-the-art LLMs, including commercial and open source systems. We find that instrumental self-reasoning ability emerges only in the most capable frontier models and that it is highly context-dependent. No model passes the the most difficult versions of our evaluations, hence our evaluation can be used to measure increases in instrumental self-reasoning ability in future models. We open-source our evaluations at https://github.com/kaifronsdal/Self-Reasoning-Evals.
    **URL:** http://arxiv.org/abs/2412.03904v1
    **Title:** Logic-Enhanced Language Model Agents for Trustworthy Social Simulations
    **Authors:** Agnieszka Mensfelt, Kostas Stathis, Vince Trencsenyi
    **Published:** 2024
    **Summary:** We introduce the Logic-Enhanced Language Model Agents (LELMA) framework, a novel approach to enhance the trustworthiness of social simulations that utilize large language models (LLMs). While LLMs have gained attention as agents for simulating human behaviour, their applicability in this role is limited by issues such as inherent hallucinations and logical inconsistencies. LELMA addresses these challenges by integrating LLMs with symbolic AI, enabling logical verification of the reasoning generated by LLMs. This verification process provides corrective feedback, refining the reasoning output. The framework consists of three main components: an LLM-Reasoner for producing strategic reasoning, an LLM-Translator for mapping natural language reasoning to logic queries, and a Solver for evaluating these queries. This study focuses on decision-making in game-theoretic scenarios as a model of human interaction. Experiments involving the Hawk-Dove game, Prisoner's Dilemma, and Stag Hunt highlight the limitations of state-of-the-art LLMs, GPT-4 Omni and Gemini 1.0 Pro, in producing correct reasoning in these contexts. LELMA demonstrates high accuracy in error detection and improves the reasoning correctness of LLMs via self-refinement, particularly in GPT-4 Omni.
    **URL:** http://arxiv.org/abs/2408.16081v1
    **Title:** Laying the Foundation First? Investigating the Generalization from Atomic Skills to Complex Reasoning Tasks
    **Authors:** Yuncheng Huang, Qianyu He, Yipei Xu, Jiaqing Liang, Yanghua Xiao
    **Published:** 2024
    **Summary:** Current language models have demonstrated their capability to develop basic reasoning, but struggle in more complicated reasoning tasks that require a combination of atomic skills, such as math word problem requiring skills like arithmetic and unit conversion. Previous methods either do not improve the inherent atomic skills of models or not attempt to generalize the atomic skills to complex reasoning tasks. In this paper, we first propose a probing framework to investigate whether the atomic skill can spontaneously generalize to complex reasoning tasks. Then, we introduce a hierarchical curriculum learning training strategy to achieve better skill generalization. In our experiments, we find that atomic skills can not spontaneously generalize to compositional tasks. By leveraging hierarchical curriculum learning, we successfully induce generalization, significantly improve the performance of open-source LMs on complex reasoning tasks. Promisingly, the skill generalization exhibit effective in cross-dataset and cross-domain scenarios. Complex reasoning can also help enhance atomic skills. Our findings offer valuable guidance for designing better training strategies for complex reasoning tasks.
    **URL:** http://arxiv.org/abs/2403.09479v1
    **Title:** Reasoning in Large Language Models: A Geometric Perspective
    **Authors:** Romain Cosentino, Sarath Shekkizhar
    **Published:** 2024
    **Summary:** The advancement of large language models (LLMs) for real-world applications hinges critically on enhancing their reasoning capabilities. In this work, we explore the reasoning abilities of large language models (LLMs) through their geometrical understanding. We establish a connection between the expressive power of LLMs and the density of their self-attention graphs. Our analysis demonstrates that the density of these graphs defines the intrinsic dimension of the inputs to the MLP blocks. We demonstrate through theoretical analysis and toy examples that a higher intrinsic dimension implies a greater expressive capacity of the LLM. We further provide empirical evidence linking this geometric framework to recent advancements in methods aimed at enhancing the reasoning capabilities of LLMs.
    **URL:** http://arxiv.org/abs/2407.02678v1
    **Title:** Enhance Reasoning Ability of Visual-Language Models via Large Language Models
    **Authors:** Yueting Yang, Xintong Zhang, Wenjuan Han
    **Published:** 2023
    **Summary:** Pre-trained visual language models (VLM) have shown excellent performance in image caption tasks. However, it sometimes shows insufficient reasoning ability. In contrast, large language models (LLMs) emerge with powerful reasoning capabilities. Therefore, we propose a method called TReE, which transfers the reasoning ability of a large language model to a visual language model in zero-shot scenarios. TReE contains three stages: observation, thinking, and re-thinking. Observation stage indicates that VLM obtains the overall information of the relative image. Thinking stage combines the image information and task description as the prompt of the LLM, inference with the rationals. Re-Thinking stage learns from rationale and then inference the final result through VLM.
    **URL:** http://arxiv.org/abs/2305.13267v1
    **Title:** Gemini in Reasoning: Unveiling Commonsense in Multimodal Large Language Models
    **Authors:** Yuqing Wang, Yun Zhao
    **Published:** 2023
    **Summary:** The burgeoning interest in Multimodal Large Language Models (MLLMs), such as OpenAI's GPT-4V(ision), has significantly impacted both academic and industrial realms. These models enhance Large Language Models (LLMs) with advanced visual understanding capabilities, facilitating their application in a variety of multimodal tasks. Recently, Google introduced Gemini, a cutting-edge MLLM designed specifically for multimodal integration. Despite its advancements, preliminary benchmarks indicate that Gemini lags behind GPT models in commonsense reasoning tasks. However, this assessment, based on a limited dataset (i.e., HellaSWAG), does not fully capture Gemini's authentic commonsense reasoning potential. To address this gap, our study undertakes a thorough evaluation of Gemini's performance in complex reasoning tasks that necessitate the integration of commonsense knowledge across modalities. We carry out a comprehensive analysis of 12 commonsense reasoning datasets, ranging from general to domain-specific tasks. This includes 11 datasets focused solely on language, as well as one that incorporates multimodal elements. Our experiments across four LLMs and two MLLMs demonstrate Gemini's competitive commonsense reasoning capabilities. Additionally, we identify common challenges faced by current LLMs and MLLMs in addressing commonsense problems, underscoring the need for further advancements in enhancing the commonsense reasoning abilities of these models.
    **URL:** http://arxiv.org/abs/2312.17661v1
    **Title:** Enhance Reasoning Ability of Visual-Language Models via Large Language Models
    **Authors:** Yueting Yang, Xintong Zhang, Wenjuan Han
    **Published:** 2023
    **Summary:** Pre-trained visual language models (VLM) have shown excellent performance in image caption tasks. However, it sometimes shows insufficient reasoning ability. In contrast, large language models (LLMs) emerge with powerful reasoning capabilities. Therefore, we propose a method called TReE, which transfers the reasoning ability of a large language model to a visual language model in zero-shot scenarios. TReE contains three stages: observation, thinking, and re-thinking. Observation stage indicates that VLM obtains the overall information of the relative image. Thinking stage combines the image information and task description as the prompt of the LLM, inference with the rationals. Re-Thinking stage learns from rationale and then inference the final result through VLM.
    **URL:** http://arxiv.org/abs/2305.13267v1
    

    2025-01-07 09:25:46,188 - INFO - HTTP Request: POST https://api.chatanywhere.tech/chat/completions "HTTP/1.1 200 OK"
    2025-01-07 09:25:46,342 - INFO - Search and summary completed.
    


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
   - **Download:** [arXiv](http://arxiv.org/abs/2410.13080v1)

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
