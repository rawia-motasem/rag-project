import os
import streamlit as st
from openai import OpenAI

def get_client():
    """
    يتحقق من وجود المفتاح في Streamlit Secrets أو متغيرات البيئة
    ويقوم بتهيئة Client الخاص بـ OpenRouter أو OpenAI.
    """
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    # التحقق من أن المفتاح ليس النص التوضيحي الافتراضي
    if openrouter_key and "your-actual" not in openrouter_key:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            default_headers={
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Green Future Climate RAG",
            }
        )

    if openai_key and "your-actual" not in openai_key:
        return OpenAI(api_key=openai_key)

    raise ValueError(
        "مفتاح الـ API الموجود في Secrets غير صحيح أو لا يزال يحتوي على قيمة توضيحية وهمية. يرجى إدخال مفتاح حقيقي من OpenRouter أو OpenAI."
    )


def retrieve_documents(query, n_results=3):
    """
    استرجاع النصوص المرجعية المقترنة بالمستندات المناخية.
    """
    sample_corpus = [
        {
            "metadata": {"title": "IPCC AR6 Synthesis Report", "status": "CURRENT"},
            "text": "Global greenhouse gas emissions must decline by 43% by 2030 relative to 2019 levels to limit global warming to 1.5°C.",
            "distance": 0.1245,
        },
        {
            "metadata": {"title": "COP29 Finance Agreement (NCQG)", "status": "CURRENT"},
            "text": "The New Collective Quantified Goal (NCQG) sets a target for climate finance, replacing the previous $100 billion per year target.",
            "distance": 0.1892,
        },
        {
            "metadata": {"title": "Global Renewables Status Report", "status": "CURRENT"},
            "text": "Battery storage technology expanded rapidly, helping integrate variable solar and wind power into national energy grids.",
            "distance": 0.2310,
        },
        {
            "metadata": {"title": "Paris Agreement Article 2", "status": "OUTDATED"},
            "text": "The core objective of the Paris Agreement is keeping global temperature rise well below 2.0°C and pursuing efforts for 1.5°C.",
            "distance": 0.3105,
        },
    ]

    return sample_corpus[:n_results]


def generate_answer(query, n_results=3):
    """
    توليد إجابة موثقة بالمصادر بناءً على الـ Retrieval.
    """
    sources = retrieve_documents(query, n_results=n_results)

    context_text = "\n\n".join(
        [f"Source [{i+1}] ({doc['metadata']['title']}):\n{doc['text']}" for i, doc in enumerate(sources)]
    )

    system_prompt = (
        "You are 'Green Future', a climate policy assistant. "
        "Answer the question based strictly on the provided context below. "
        "If the answer cannot be found in the context, clearly state that you do not have enough information."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    client = get_client()
    model_name = st.secrets.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        return answer, sources

    except Exception as err:
        if "401" in str(err) or "User not found" in str(err):
            raise RuntimeError(
                "خطأ 401: مفتاح OpenRouter غير صحيح أو منتهي الصلاحية. يرجى إنشاء مفتاح جديد من openrouter.ai/keys وتحديث Secrets في Streamlit Cloud."
            ) from err
        raise err
