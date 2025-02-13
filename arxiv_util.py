import arxiv
from datetime import datetime
from typing import List, Optional
import asyncio


def get_arxiv_results(
    query: str,
    max_results: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    categories: Optional[List[str]] = None,
    abstract: bool = False,
) -> list:
    """
    高级arXiv论文检索

    参数:
        query: str - 搜索关键词
        max_results: int - 最大返回结果数
        start_date: datetime - 开始日期 (可选)
        end_date: datetime - 结束日期 (可选)
        categories: List[str] - arXiv分类目录 (可选)
        abstract: bool - 是否在摘要中搜索

    分类示例:
        cs.AI - 人工智能
        cs.CL - 计算语言学
        cs.LG - 机器学习
        cs.CV - 计算机视觉
        stat.ML - 统计学习
    """
    # 构建高级搜索查询
    advanced_query = query

    # 添加日期范围
    if start_date:
        advanced_query += (
            f" AND submittedDate:[{start_date.strftime('%Y%m%d')}000000 TO "
        )
        if end_date:
            advanced_query += f"{end_date.strftime('%Y%m%d')}235959]"
        else:
            advanced_query += "now]"

    # 添加分类目录
    if categories:
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        advanced_query += f" AND ({cat_query})"

    # 是否在摘要中搜索
    if abstract:
        advanced_query = f"abs:{advanced_query}"
    else:
        advanced_query = f"all:{advanced_query}"

    # 创建搜索对象
    client = arxiv.Client()
    search = arxiv.Search(
        query=advanced_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    try:
        results = list(client.results(search))
        return results
    except Exception as e:
        print(f"搜索出错: {str(e)}")
        return []


async def async_get_arxiv_results(
    query: str,
    max_results: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    categories: Optional[List[str]] = None,
    abstract: bool = False,
) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        get_arxiv_results,
        query,
        max_results,
        start_date,
        end_date,
        categories,
        abstract,
    )


async def main():
    results = await async_get_arxiv_results(
        query="machine learning",
        max_results=10,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 12, 31),
        categories=["cs.AI", "cs.LG"],
        abstract=True,
    )
    for result in results:
        print(result.entry_id, result.title)


# 运行异步函数
if __name__ == "__main__":
    asyncio.run(main())
