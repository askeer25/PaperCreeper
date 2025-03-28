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

### 从命令行使用

运行 Web UI 版本：

```bash
streamlit run webui.py
```

然后在浏览器中访问显示的本地URL（通常为 `http://localhost:8501`）使用图形界面。

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