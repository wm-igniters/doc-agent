"""
Response generator using LLM providers for answer synthesis.

Takes retrieved context and generates a helpful response
with proper citations and formatting.

Features:
- Multi-provider support (Anthropic, OpenAI, Ollama)
- Prompt caching (where supported)
- Streaming responses
"""

import logging
from typing import Any, AsyncGenerator, Optional

from src.api.models import Source, Video
from src.config.settings import get_settings
from src.core.providers import get_llm_provider, BaseLLMProvider
from src.core.providers.base import StreamResult, TokenUsage
from src.core.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are the **WaveMaker Documentation Assistant**, an expert on the WaveMaker low-code development platform. Stay friendly, concise, and solution-oriented while answering the user's question using only the provided context.

## 🎯 Mission
- Deliver a confident, actionable answer grounded in the supplied documentation. Do **not** invent information beyond that context.

## 🧭 Response Flow
Use these sections when they add value; skip any that are irrelevant.
1. **Direct Answer** – 1‑2 sentences that address the question immediately.
2. **Key Takeaways** – 2‑3 short bullets summarizing the most important points or decisions.
3. **Details** – Organize supporting information into logical sub-headings. Synthesize the context into one coherent narrative; avoid repeating near-identical sentences from multiple sources.
4. **Steps** – When the user needs to perform actions, provide a numbered checklist with UI paths, prerequisites, and tips.
5. **Code Examples / Snippets** – Only when the context includes code. Use fenced blocks with language tags.
6. **Related Info** – Optional pointers to adjacent features, caveats, or troubleshooting notes.

## ✅ Citations & Sources
- Cite evidence inline using numbered references, e.g. [1]. Citation numbers are resolved by the frontend.
- Each number must map to a source in the provided context; bundle multiple supporting sources like [1][2].
- Add a short **Further Reading** list at the end when there are helpful links: `- [Title][n]`.

## 📺 Video Recommendations
When videos are provided, add a **Related Videos** section using markdown links: `[Video Title](URL)`. Include only videos that clearly help the user.

## 🤖 Style & Formatting
- Keep the tone friendly and practical.
- Highlight UI elements, APIs, and key terms with **bold** or `code`.
- Use tables when they clarify comparisons or configuration options.
- Numbered lists for sequential steps; bullets for unordered lists.

## ⚠️ Knowledge Limits
If the context lacks the required information:
- Say: “I don’t have specific information about this in the documentation.”
- Suggest where the user might look next (other features, docs sections) without guessing.

Before responding, double-check:
- Direct answer appears first.
- Every factual statement is cited.
- Steps are clear and ordered.
- Videos and further reading are included only when relevant."""


class ResponseGenerator:
    """
    Generates responses using LLM providers with retrieved context.
    
    Features:
    - Multi-provider support (Anthropic, OpenAI, Ollama)
    - Streaming support for real-time responses
    - Structured output with sources and videos
    """

    def __init__(self):
        self.settings = get_settings()
        self._provider: Optional[BaseLLMProvider] = None

    def _get_provider(self) -> BaseLLMProvider:
        """Get or create LLM provider."""
        if self._provider is None:
            self._provider = get_llm_provider(self.settings)
            logger.info(f"Using LLM provider: {self.settings.ai_provider} ({self._provider.model_name})")
        return self._provider

    def _format_context(
        self,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Format retrieved documents and videos into context string."""
        parts = ["## 📚 Documentation Context\n"]

        for i, doc in enumerate(documents, 1):
            section_info = f" > {doc.section}" if doc.section else ""
            parts.append(f"### [{i}] {doc.title}{section_info}")
            parts.append(f"📎 URL: {doc.url}")
            parts.append("---")
            parts.append(doc.content)
            parts.append("")

        if videos:
            parts.append("\n## 🎬 Related Videos (supplementary)\n")
            for video in videos:
                duration = f" ({video.get('duration', 'N/A')})" if video.get('duration') else ""
                title = video.get('title', 'Video')
                link = video.get('link', '')
                # Academy MCP uses "link" field, not "url"
                parts.append(f"📺 [{title}]({link}){duration}")
            parts.append("")
            parts.append("*(Note: Include these video links in your response as clickable markdown links. You cannot watch them, but users can.)*")

        return "\n".join(parts)

    def _build_user_prompt(
        self,
        query: str,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Build the complete user prompt with context and query."""
        context = self._format_context(documents, videos)

        return f"""{context}

---

## ❓ Question

{query}

---

**Instructions:** Answer the question based on the documentation context above.
- Start with a direct answer
- Use proper markdown formatting
- Cite sources using [1], [2], etc. inline
- Include code examples if relevant (with language tags)"""

    async def generate(
        self,
        query: str,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Generate a complete response (non-streaming).

        Returns:
            Dict with answer, sources, and videos
        """
        provider = self._get_provider()
        user_prompt = self._build_user_prompt(query, documents, videos)

        result = await provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

        sources = self._extract_sources(documents)
        video_list = self._extract_videos(videos) if videos else []

        response = {
            "answer": result.text,
            "sources": sources,
            "videos": video_list,
        }
        
        # Include token usage if available
        if result.usage:
            response["usage"] = {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        
        return response

    async def generate_stream(
        self,
        query: str,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate a streaming response.

        Yields:
            Dict chunks with type and content
        """
        provider = self._get_provider()
        user_prompt = self._build_user_prompt(query, documents, videos)

        usage: TokenUsage | None = None
        
        try:
            async for chunk in provider.generate_stream(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_prompt,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            ):
                # StreamResult contains usage at end of stream
                if isinstance(chunk, StreamResult):
                    usage = chunk.usage
                else:
                    yield {"type": "text", "content": chunk}

            # After text is complete, yield sources and videos
            sources = self._extract_sources(documents)
            yield {"type": "sources", "sources": [s.model_dump() for s in sources]}

            if videos:
                video_list = self._extract_videos(videos)
                yield {"type": "videos", "videos": [v.model_dump() for v in video_list]}

            # Include usage in done chunk
            done_chunk = {"type": "done", "provider": self.settings.ai_provider}
            if usage:
                done_chunk["usage"] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
            yield done_chunk

        except Exception as e:
            logger.exception(f"Error during streaming generation: {e}")
            yield {"type": "error", "error": str(e)}

    def _extract_sources(self, documents: list[RetrievedDocument]) -> list[Source]:
        """Extract source citations from documents."""
        sources = []
        for i, doc in enumerate(documents, 1):
            sources.append(
                Source(
                    id=i,
                    title=doc.title,
                    url=doc.url,
                    section=doc.section,
                    relevance_score=doc.rrf_score,
                )
            )
        return sources

    def _extract_videos(self, videos: list[dict[str, Any]]) -> list[Video]:
        """
        Extract video objects from Academy MCP response.

        Academy MCP returns videos with fields:
        - title, description, moduleName, code, link (not url)
        """
        return [
            Video(
                title=v.get("title", "Video"),
                url=v.get("link", ""),  # Academy MCP uses "link" field
                duration=v.get("duration"),
            )
            for v in videos
        ]
