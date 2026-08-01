"""
01_documents.py
----------------
Step 1 of the RAG pipeline: Raw Documents.

Purpose:
    This file is responsible for ONE thing only: providing the raw text
    documents that the rest of the pipeline (preprocessing, chunking,
    embeddings, vector store, retrieval) will operate on.

    Nothing here is cleaned, chunked, or embedded yet — that happens in
    later files (02_preprocessing.py, 03_chunking.py, ...).

Topic:
    Climate policy documents (reused from the previous project), including
    both CURRENT and OUTDATED records, to demonstrate that the RAG system
    can distinguish between them using metadata.
"""

# Each document is a dictionary with:
#   - id: unique identifier
#   - title: short human-readable name
#   - status: "CURRENT" or "OUTDATED" (used later for metadata filtering)
#   - department: source department (used later for metadata filtering)
#   - text: the raw document content

RAW_DOCUMENTS = [
    {
        "id": "doc0",
        "title": "Paris Agreement Overview",
        "status": "CURRENT",
        "department": "Climate Action Department",
        "text": (
            "The Paris Agreement is a legally binding international treaty on "
            "climate change adopted in 2015. Its central aim is to limit global "
            "warming to well below 2 degrees Celsius, preferably to 1.5 degrees "
            "Celsius, compared to pre-industrial levels. To limit global warming "
            "to 1.5 degrees Celsius, greenhouse gas emissions must peak before "
            "2025 at the latest and decline 43% by 2030. Countries submit "
            "nationally determined contributions (NDCs) outlining their emission "
            "reduction targets, and these are reviewed every five years."
        ),
    },
    {
        "id": "doc1",
        "title": "Climate Economy",
        "status": "CURRENT",
        "department": "Climate Action Department",
        "text": (
            "Transitioning to a green economy opens up vast profitable "
            "opportunities in zero-carbon business. Investment in renewable "
            "energy drives long-term growth, creates millions of jobs "
            "worldwide, and reduces dependency on volatile fossil fuel markets. "
            "Sectors such as solar manufacturing, electric vehicles, and green "
            "hydrogen are attracting record levels of private and public "
            "investment."
        ),
    },
    {
        "id": "doc2",
        "title": "COP Agreements",
        "status": "CURRENT",
        "department": "Climate Action Department",
        "text": (
            "The Conference of the Parties (COP) is the main decision-making "
            "body under the UN climate framework. Recent COP agreements "
            "established a new collective climate finance goal to help "
            "developing countries adapt to climate change and transition to "
            "clean energy. This new goal replaces earlier finance targets and "
            "reflects updated estimates of the true cost of climate action."
        ),
    },
    {
        "id": "doc3",
        "title": "Renewable Energy Technology",
        "status": "CURRENT",
        "department": "Technology Department",
        "text": (
            "Advances in battery storage and solar panel efficiency have "
            "significantly lowered the cost of renewable energy over the past "
            "decade. Grid-scale battery storage now allows solar and wind power "
            "to supply electricity even when the sun is not shining or the wind "
            "is not blowing, improving the reliability of renewable energy "
            "sources."
        ),
    },
    {
        "id": "doc4",
        "title": "Old Finance Notice",
        "status": "OUTDATED",
        "department": "Historical Records Department",
        "text": (
            "The old 2020 target for climate finance aimed to mobilize 100 "
            "billion dollars annually from developed countries to support "
            "developing nations. This target has been updated and replaced by "
            "a newer, larger finance goal agreed upon in subsequent COP "
            "negotiations."
        ),
    },
]


def get_raw_documents():
    """
    Returns the list of raw documents.

    This is the function that 02_preprocessing.py will import and call.
    """
    return RAW_DOCUMENTS


if __name__ == "__main__":
    # Quick manual check when running this file directly:
    # python 01_documents.py
    docs = get_raw_documents()
    print(f"Loaded {len(docs)} raw documents:\n")
    for d in docs:
        print(f"- [{d['id']}] {d['title']} ({d['status']}, {d['department']})")
