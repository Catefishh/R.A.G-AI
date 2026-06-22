"""Command-line comparison: RAG vs no-RAG, printed side by side in the terminal.

Usage:
    python compare_cli.py "Who founded Nimbus Forge?"
    python compare_cli.py            # interactive prompt loop
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_ai import Config, RAGPipeline  # noqa: E402


def _wrap(text: str, width: int = 78) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else line
        for line in text.splitlines()
    )


def run_once(pipeline: RAGPipeline, question: str) -> None:
    comparison = pipeline.compare(question)

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    print("\n--- WITHOUT RAG (model alone) ---\n")
    print(_wrap(comparison.plain.text))
    print(f"\n[{comparison.plain.elapsed_s:.1f}s]")

    print("\n--- WITH RAG (model + retrieved context) ---\n")
    print(_wrap(comparison.rag.text))
    print(f"\n[{comparison.rag.elapsed_s:.1f}s]")

    if comparison.rag.retrieved:
        print("\nRetrieved sources:")
        for i, r in enumerate(comparison.rag.retrieved, start=1):
            print(f"  [{i}] {r.source}  (similarity {r.score:.3f})")
    else:
        print("\nRetrieved sources: none above threshold")
    print()


def main() -> None:
    print("Loading models and building the knowledge base… (first run downloads models)")
    pipeline = RAGPipeline(Config())
    n = pipeline.build_index()
    print(f"Indexed {n} chunks from {pipeline.config.docs_dir}.")

    args = [a for a in sys.argv[1:] if a.strip()]
    if args:
        run_once(pipeline, " ".join(args))
        return

    print("\nEnter a question (blank line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        run_once(pipeline, question)


if __name__ == "__main__":
    main()
