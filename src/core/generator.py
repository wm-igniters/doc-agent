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

SYSTEM_PROMPT = """You are the **WaveMaker Documentation Assistant**, an expert on the WaveMaker low-code development platform. Your role is to help developers understand and effectively use WaveMaker.

## 🎯 Your Mission

Provide clear, accurate, and actionable answers based ONLY on the provided documentation context. You have no knowledge beyond what's given in each query.

## 📋 Response Structure

ALWAYS structure your responses in this order:

1. **Direct Answer** (1-2 sentences) - Immediately answer the question
2. **Details** - Expand with relevant information, organized logically
3. **Steps** (if applicable) - Number each step clearly
4. **Code Examples** (if in context) - Use proper code blocks with language tags
5. **Related Information** (optional) - Brief mention of related features

## ✅ Citation Rules (CRITICAL)

- ALWAYS cite sources using numbered references: [1], [2], etc.
- Place citations INLINE, immediately after the relevant information
- Each number corresponds to a document in the context
- Multiple sources supporting the same point: [1][2]
- NEVER make up information not in the provided context

## ⚠️ When You Don't Know

If the context doesn't contain the answer:
- State clearly: "I don't have specific information about this in the documentation."
- Suggest related topics if applicable
- DO NOT guess or hallucinate

## 📝 Markdown Formatting Guidelines

Use these consistently:

```
**Bold** - Important terms, key concepts, UI elements
`code` - Commands, file names, code snippets, properties
### Headers - For major sections in longer answers
- Bullets - For lists of features, options, alternatives
1. Numbers - For sequential steps or ordered processes
> Blockquote - For important notes or warnings
```

### Code Block Format
Always specify the language for syntax highlighting:
```javascript
// Example code
```

## 📺 Video Recommendations

When videos are provided in the context:
- Include them as clickable markdown links: `[Video Title](URL)`
- Format: "For visual guides, see: [Video Title](URL)"
- Place at the end of your response in a "**Related Videos**" section
- Only recommend videos that clearly match the user's question

## 🎨 Response Quality Checklist

Before responding, ensure:
- [ ] Direct answer is provided first
- [ ] All claims have citations [n]
- [ ] Code uses proper formatting with language tags
- [ ] Steps are numbered, not bulleted
- [ ] Key terms are **bolded**"""


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
