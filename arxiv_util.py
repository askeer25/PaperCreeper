import arxiv
import datetime


def get_arxiv_results(query, max_results):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = client.results(search)
    print(results)
    return list(results)


def get_arxiv_message(result):
    summary = result.summary.replace("\n", " ")
    authors = ", ".join([author.name for author in result.authors])
    published = result.published.year
    message = (
        f"**Title:** {result.title}\n"
        f"**Authors:** {authors}\n"
        f"**Published:** {published}\n"
        f"**Summary:** {summary}\n"
        f"**URL:** {result.entry_id}"
    )
    # print(message)
    return message
