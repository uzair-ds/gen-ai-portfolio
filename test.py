from pathlib import Path

structure = [
    "qa_agent/app/api.py",
    "qa_agent/app/config.py",

    "qa_agent/app/core/base_agent.py",
    "qa_agent/app/core/orchestrator.py",
    "qa_agent/app/core/memory.py",

    "qa_agent/app/ingestion/pdf_ingestor.py",

    "qa_agent/app/retrieval/query_planner.py",
    "qa_agent/app/retrieval/query_rewriter.py",
    "qa_agent/app/retrieval/retriever.py",
    "qa_agent/app/retrieval/reranker.py",

    "qa_agent/app/generation/context_compressor.py",
    "qa_agent/app/generation/answer_generator.py",

    "qa_agent/app/guardrails/sufficiency_guard.py",
    "qa_agent/app/guardrails/citation_guard.py",

    "qa_agent/app/storage/vector_store.py",

    "qa_agent/data/uploads/",

    "qa_agent/requirements.txt",
    "qa_agent/README.md",
]

for path in structure:
    p = Path(path)
    if path.endswith("/"):
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)

# Create __init__.py in all app subfolders
for folder in Path("qa_agent/app").rglob("*"):
    if folder.is_dir():
        (folder / "__init__.py").touch(exist_ok=True)

print("✅ qa_agent project structure created successfully")
