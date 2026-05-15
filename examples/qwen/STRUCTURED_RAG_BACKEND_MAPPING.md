# Structured RAG Backend Mapping

This note records how the Qwen examples' structured policy chunks should map
into the application backend once the example pipeline is promoted.

## Example Chunk Fields

`policy_structure.StructuredChunk` emits:

- `doc_id`: source document import id.
- `chunk_id`: stable chunk id for the structured block.
- `chunk_order_index`: order within the source document.
- `file_name`: display file name.
- `page_idx`: zero-based page index when available.
- `chapter_title`: current `第...章` heading.
- `section_title`: current `第...节` heading.
- `article_no`: current `第...条` number when available.
- `content_type`: `article`, `table`, or `section_text`.
- `content`: original text for the structured block.
- `source_block_range`: normalized parser block range.

The LightRAG example stores these as metadata lines inside `index_text` so that
the existing KV/vector storage can be used without schema changes.

## Backend Field Mapping

Recommended mapping for `code/backend`:

- `documents.id` or import job document id <- `doc_id`
- `document_chunks.id` <- `chunk_id`
- `document_chunks.content` <- `content`
- `document_chunks.chunk_order_index` <- `chunk_order_index`
- `document_chunks.page_no` <- `page_idx + 1`
- `document_chunks.section_path` <- `chapter_title / section_title`
- `document_chunks.clause_no` <- `article_no`
- `document_chunks.metadata.content_type` <- `content_type`
- `document_chunks.metadata.file_name` <- `file_name`
- `document_chunks.metadata.source_block_range` <- `source_block_range`

`Retriever.search()` already has the output shape needed by QA:

- `page_no`
- `section_path`
- `clause_no`

The answer prompt should cite these fields directly instead of asking the LLM to
infer citations from free text.

## Migration Notes

1. Keep the examples' `StructuredChunk` as the reference shape.
2. Add persistent chunk metadata columns or JSONB fields in backend storage.
3. Compute embeddings from `content` plus a short metadata prefix containing
   section path and clause number.
4. Use raw `content` for final answer context.
5. Use metadata for citations and filters.

## Retrieval Behavior

For policy QA, prefer:

1. vector or hybrid retrieval over structured chunks;
2. rerank using query + `section_path + clause_no + content`;
3. neighbor expansion by `chunk_order_index` within the same document only when
   the top chunk is an article/table continuation;
4. final answer context limited to the top reranked chunks with metadata.

## Recommended Import Strategy

For policy QA systems that prioritize grounded answers and stable citations,
structured chunks should be indexed in a retrieval-first mode:

- write `full_docs`, `text_chunks`, `vdb_chunks`, and `doc_status`;
- treat entity / relationship extraction as optional enhancement;
- do not let KG extraction failures block the document from becoming queryable.
