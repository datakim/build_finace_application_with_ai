import openai
from my_vectorstore import VectorStore

def retrieve_relevant_docs(query: str, k=2):                     #A
    """Fetch top 'k' matching policy docs from a BFSI vector store."""
    return VectorStore.similarity_search(query, k=k)             

def build_prompt(user_query: str, docs: list):                    #B
    """Combine user query and doc snippets into a single LLM prompt."""
    context_snippets = "\n".join([d.page_content for d in docs])
    prompt = f"""
You are a BFSI generative AI assistant. 
Use the official references below to answer accurately.

Official References:
{context_snippets}

User Query:
{user_query}

Final Output:
"""
    return prompt

def answer_with_rag(user_query: str):
    docs = retrieve_relevant_docs(user_query)                     #C
    final_prompt = build_prompt(user_query, docs)
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",                                     
      messages=[{"role": "system",                               
                 "content": "Adhere to BFSI compliance."},
                {"role": "user", "content": final_prompt}],
      temperature=0.0
    )
    return response["choices"][0]["message"]["content"]

query_text = "What's the latest LTV ratio cap for mortgages in region X?"  
answer = answer_with_rag(query_text)                               #D
print("LLM + RAG answer:\n", answer)                               #E
