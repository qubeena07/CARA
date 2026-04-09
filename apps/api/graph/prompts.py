GRADE_DOCUMENT_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.
Document: {document}
Question: {question}
Give a binary score 'relevant' or 'irrelevant'."""

GENERATE_ANSWER_PROMPT = """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
Context: {context}
Question: {question}
Answer:"""

REWRITE_QUERY_PROMPT = """You are a query rewriter. Rewrite the following query to improve retrieval.
Original query: {query}
Rewritten query:"""

HALLUCINATION_CHECK_PROMPT = """You are checking if an answer is grounded in the provided context.
Context: {context}
Answer: {answer}
Is the answer grounded in the context? Respond with 'yes' or 'no'."""
