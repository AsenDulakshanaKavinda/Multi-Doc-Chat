
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

""""
This prompt tells the model to rephrase the user’s query so it becomes context-independent (standalone).
If the user’s question depends on prior messages, this step rewrites it so it makes full sense by itself.
"""
contextualize_question_prompt = ChatPromptTemplate.chat_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

"""
This step ensures grounded responses — the model answers only using retrieved data, not from its own parametric memory.
This is crucial for factual consistency and trustworthy RAG behavior.
"""
context_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# central dictionary to register prompts
PREOMPT_REGISTRY = {
    "contextualize_question" : contextualize_question_prompt,
    "context_question": context_question_prompt
}











