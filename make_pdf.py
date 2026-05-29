from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("test_en.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = [
    Paragraph("Introduction to Machine Learning", styles["Title"]),
    Paragraph("Machine learning is a branch of artificial intelligence. Systems learn from data instead of following explicit rules. There are three main types: supervised learning, unsupervised learning, and reinforcement learning.", styles["Normal"]),
    Paragraph("What is an Embedding", styles["Heading1"]),
    Paragraph("An embedding is a vector of numbers that encodes the meaning of text. Semantically similar texts have similar vectors. The distance between vectors is measured using cosine similarity.", styles["Normal"]),
    Paragraph("What is RAG", styles["Heading1"]),
    Paragraph("RAG stands for Retrieval-Augmented Generation. It combines search over a knowledge base with text generation by a language model. Documents are split into chunks, embedded, and stored in a vector database. At query time the most relevant chunks are retrieved and passed to the LLM.", styles["Normal"]),
    Paragraph("Context Window and Tokens", styles["Heading1"]),
    Paragraph("A token is the basic unit of text processed by a language model. One token is roughly 4 characters in English. The context window is the maximum number of tokens a model can process at once.", styles["Normal"]),
]
doc.build(story)
print("Created test_en.pdf")