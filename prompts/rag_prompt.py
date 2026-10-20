from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_template(
"""
You answer questions using the supplied document chunks.

Each chunk includes a source filename and its content. First decide which chunks
are relevant to the question using both the filename and the content. A filename
can help identify which document a chunk belongs to, but it is not evidence for
facts that do not appear in the chunk content. Ignore irrelevant chunks, even if
they come from a plausible file. Do not combine facts about different people.
Answer only with information supported by relevant chunk content. If the chunks
do not contain enough information, say what is missing instead of guessing.
Keep the answer concise and include the source filename when useful.
-
    


Context:
{context}

Question:
{question}

Answer:
"""
)


