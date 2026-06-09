"""
Run this ONCE to parse the PPT file and build the FAISS vector index.

Command:
python store_index.py
"""

from src.ppt_parser import extract_projects_from_pptx
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

DATA_FOLDER = "data"
PPT_FILE = "Dataset_project_repository.pptx"
FAISS_INDEX_PATH = "faiss_index"


def build_index():

    # Full PPT path
    pptx_path = os.path.join(DATA_FOLDER, PPT_FILE)

    if not os.path.exists(pptx_path):
        print(f" PPT file not found: {pptx_path}")
        return

    # Step 1 — Extract projects from PPT
    projects = extract_projects_from_pptx(pptx_path)

    if not projects:
        print(" No projects extracted from PPT.")
        return

    print(f"Extracted {len(projects)} projects")

    # Step 2 — Convert projects into LangChain Documents
    docs = []

    for p in projects:

        doc = Document(
            page_content=p["combined_text"],
            metadata={
                "project_id": p["project_id"],
                "project_title": p["project_title"],
                "domain": p["domain"],
                "project_type": p["project_type"],
                "tech_stack": p["tech_stack"],
                "team_size": p["team_size"],
                "duration": p["duration"],
                "slide_numbers": str(p["slide_numbers"])
            }
        )

        docs.append(doc)

    print(f"Created {len(docs)} document chunks")

    # Step 3 — Create Embeddings
    print("🔄 Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4 — Build FAISS Index
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    # Step 5 — Save FAISS Index
    vectorstore.save_local(FAISS_INDEX_PATH)

    print(f"FAISS index saved to '{FAISS_INDEX_PATH}/'")
    print("🚀 You can now run:")
    print("streamlit run app.py")


if __name__ == "__main__":
    build_index()