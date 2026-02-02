"""
Document indexer for WaveMaker documentation.

Orchestrates:
1. Clone/pull documentation repository
2. Discover markdown files
3. Parse and chunk documents
4. Generate embeddings
5. Upsert to Qdrant
"""

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from git import Repo

from src.config.settings import get_settings
from src.core.embedder import Embedder
from src.core.retriever import HybridRetriever
from src.indexer.chunker import Chunk, SemanticChunker
from src.indexer.parser import MarkdownParser, ParsedDocument

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Indexes WaveMaker documentation from GitHub into Qdrant.
    """

    def __init__(self):
        self.settings = get_settings()
        self.parser = MarkdownParser()
        self.chunker = SemanticChunker()
        self.embedder = Embedder()
        self.retriever = HybridRetriever()

    async def index(
        self,
        branch: Optional[str] = None,
        force: bool = False,
        local_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Index documentation from GitHub.

        Args:
            branch: Git branch to index (defaults to settings)
            force: Force full reindex
            local_path: Optional local path instead of cloning

        Returns:
            Indexing statistics
        """
        start_time = time.time()
        branch = branch or self.settings.docs_branch
        errors = []

        logger.info(f"Starting documentation indexing (branch: {branch})")

        try:
            # Step 1: Get documentation files
            if local_path:
                docs_path = Path(local_path)
                logger.info(f"Using local path: {docs_path}")
            else:
                docs_path = await self._clone_repo(branch)
                logger.info(f"Cloned repo to: {docs_path}")

            # Step 2: Discover markdown files
            md_files = self._discover_files(docs_path)
            logger.info(f"Discovered {len(md_files)} markdown files")

            if not md_files:
                return {
                    "status": "completed",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "duration_seconds": time.time() - start_time,
                    "errors": ["No markdown files found"],
                }

            # Step 3: Parse documents
            documents = []
            for file_path in md_files:
                try:
                    doc = self.parser.parse_file(file_path)
                    if doc and self.parser.should_index(doc):
                        documents.append(doc)
                except Exception as e:
                    errors.append(f"Parse error ({file_path.name}): {str(e)}")

            logger.info(f"Parsed {len(documents)} documents")

            # Step 4: Chunk documents
            all_chunks = []
            for doc in documents:
                try:
                    chunks = self.chunker.chunk_document(doc)
                    all_chunks.extend(chunks)
                except Exception as e:
                    errors.append(f"Chunk error ({doc.file_path}): {str(e)}")

            logger.info(f"Created {len(all_chunks)} chunks")

            if not all_chunks:
                return {
                    "status": "completed",
                    "documents_processed": len(documents),
                    "chunks_created": 0,
                    "duration_seconds": time.time() - start_time,
                    "errors": errors or ["No chunks created"],
                }

            # Step 5: Generate embeddings (batched)
            logger.info("Generating embeddings...")
            contents = [chunk.content for chunk in all_chunks]
            embeddings = await self.embedder.embed_documents_batch(
                contents,
                batch_size=32,
            )

            # Step 6: Prepare points for Qdrant
            points = []
            for chunk, embedding in zip(all_chunks, embeddings):
                # Generate sparse vector (already has int indices from embedder)
                sparse_vector = self.embedder.generate_sparse_vector(chunk.content)

                point = {
                    "id": chunk.id,
                    "dense_vector": embedding.tolist(),
                    "sparse_vector": sparse_vector,  # Already int-keyed from embedder
                    "payload": {
                        "content": chunk.content,
                        "title": chunk.title,
                        "section": chunk.section,
                        "url": chunk.url,
                        "source": "docs",
                        "doc_type": chunk.doc_type,
                        "has_code": chunk.has_code,
                        "code_language": chunk.code_language,
                        "word_count": chunk.word_count,
                        "file_path": chunk.file_path,
                        **chunk.metadata,
                    },
                }
                points.append(point)

            # Step 7: Index to Qdrant
            logger.info(f"Indexing {len(points)} points to Qdrant...")

            # Delete existing collection if force reindex
            if force:
                await self.retriever.delete_collection()

            # Ensure collection exists
            await self.retriever.ensure_collection(
                embedding_dim=self.embedder.embedding_dimension,
            )

            # Batch upsert
            batch_size = 50  # Reduced batch size to prevent timeouts
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await self.retriever.upsert_batch(batch)
                logger.debug(f"Upserted batch {i // batch_size + 1}")

            duration = time.time() - start_time
            logger.info(
                f"Indexing complete: {len(documents)} docs, "
                f"{len(points)} chunks in {duration:.1f}s"
            )

            return {
                "status": "completed",
                "documents_processed": len(documents),
                "chunks_created": len(points),
                "duration_seconds": round(duration, 2),
                "errors": errors,
            }

        except Exception as e:
            logger.exception(f"Indexing failed: {e}")
            return {
                "status": "failed",
                "documents_processed": 0,
                "chunks_created": 0,
                "duration_seconds": time.time() - start_time,
                "errors": [str(e)] + errors,
            }

    async def _clone_repo(self, branch: str) -> Path:
        """
        Clone the documentation repository.
        """
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="wm-docs-"))

        logger.info(f"Cloning {self.settings.docs_repo_url} (branch: {branch})...")

        # Clone in a thread pool to not block async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: Repo.clone_from(
                self.settings.docs_repo_url,
                temp_dir,
                branch=branch,
                depth=1,  # Shallow clone
            ),
        )

        return temp_dir

    def _discover_files(self, docs_path: Path) -> list[Path]:
        """
        Discover all markdown files to index.
        """
        files = []

        # Look for docs directory
        docs_dirs = [
            docs_path / "docs",           # Standard Docusaurus
            docs_path / "learn" / "docs", # WaveMaker structure
            docs_path,                    # Root fallback
        ]

        docs_dir = None
        for d in docs_dirs:
            if d.exists() and d.is_dir():
                docs_dir = d
                break

        if not docs_dir:
            logger.warning("Could not find docs directory")
            return []

        # Find all .md and .mdx files
        for pattern in ["**/*.md", "**/*.mdx"]:
            for file_path in docs_dir.glob(pattern):
                # Skip files in excluded directories
                if self._should_skip_file(file_path):
                    continue
                files.append(file_path)

        return files

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path).lower()

        # Skip patterns
        skip_patterns = [
            "/node_modules/",
            "/_category_.json",
            "/versioned_docs/",  # Skip old versions for now
            "/.docusaurus/",
            "/build/",
        ]

        for pattern in skip_patterns:
            if pattern in path_str:
                return True

        # Skip files starting with underscore (except _category_.json)
        if file_path.name.startswith("_") and not file_path.name.endswith(".json"):
            return True

        return False


async def run_indexer(
    branch: Optional[str] = None,
    force: bool = False,
    local_path: Optional[str] = None,
):
    """
    CLI entry point for running the indexer.
    """
    indexer = DocumentIndexer()
    result = await indexer.index(
        branch=branch,
        force=force,
        local_path=local_path,
    )

    print("\n📚 Indexing Results:")
    print(f"   Status: {result['status']}")
    print(f"   Documents: {result['documents_processed']}")
    print(f"   Chunks: {result['chunks_created']}")
    print(f"   Duration: {result['duration_seconds']:.1f}s")

    if result['errors']:
        print(f"\n⚠️  Errors ({len(result['errors'])}):")
        for err in result['errors'][:10]:
            print(f"   - {err}")
        if len(result['errors']) > 10:
            print(f"   ... and {len(result['errors']) - 10} more")

    return result
