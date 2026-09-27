"""Bounded parent, child and neighboring chunk context expansion."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Literal, Protocol, Sequence

from config import Settings
from ingestion.chunker import RetrievalChunk
from ingestion.hierarchy import HierarchyNode
from retrieval.common import ContextEvidence, RetrievalHit


class ContextRepository(Protocol):
	"""Storage operations required for hierarchy-aware chunk expansion."""

	def get_chunks(self, document_id: str) -> tuple[RetrievalChunk, ...]: ...

	def get_hierarchy(self, document_id: str) -> tuple[HierarchyNode, ...]: ...


@dataclass(frozen=True)
class _ContextCandidate:
	chunk: RetrievalChunk
	relation: Literal["parent", "child", "neighbor"]
	priority: tuple[int, int, int]


def expand_context(
	candidates: Sequence[RetrievalHit],
	repository: ContextRepository,
	*,
	max_context_chunks: int | None = None,
	max_context_characters: int | None = None,
	settings: Settings | None = None,
) -> tuple[RetrievalHit, ...]:
	"""Attach bounded, source-traceable same-document hierarchy/neighbor context.

	Parent chunks are preferred, followed by explicit neighboring chunks and
	then descendant chunks. Context is deduplicated by chunk ID and normalized
	text across the result set, and never replaces the ranked candidate itself.
	"""
	chunk_limit = (
		max_context_chunks
		if max_context_chunks is not None
		else (settings.context_max_chunks if settings else 2)
	)
	character_limit = (
		max_context_characters
		if max_context_characters is not None
		else (settings.context_max_characters if settings else 4000)
	)
	if chunk_limit < 0:
		raise ValueError("max_context_chunks must be non-negative")
	if character_limit < 0:
		raise ValueError("max_context_characters must be non-negative")

	unique_candidates: list[RetrievalHit] = []
	seen_selected_ids: set[str] = set()
	seen_text = set()
	for hit in candidates:
		if hit.chunk_id in seen_selected_ids:
			continue
		seen_selected_ids.add(hit.chunk_id)
		unique_candidates.append(hit)
		seen_text.add(_text_identity(hit.text))

	corpus_cache: dict[str, tuple[tuple[RetrievalChunk, ...], dict[str, HierarchyNode]]] = {}
	expanded: list[RetrievalHit] = []
	for hit in unique_candidates:
		if hit.document_id not in corpus_cache:
			chunks = repository.get_chunks(hit.document_id)
			nodes = repository.get_hierarchy(hit.document_id)
			corpus_cache[hit.document_id] = (
				chunks,
				{node.node_id: node for node in nodes},
			)
		document_chunks, nodes_by_id = corpus_cache[hit.document_id]
		options = _related_chunks(hit.chunk, document_chunks, nodes_by_id)
		contexts: list[ContextEvidence] = []
		used_characters = 0
		for option in options:
			if len(contexts) >= chunk_limit:
				break
			chunk = option.chunk
			if chunk.chunk_id in seen_selected_ids:
				continue
			identity = _text_identity(chunk.text)
			if not identity or identity in seen_text:
				continue
			text_size = len(chunk.text)
			if used_characters + text_size > character_limit:
				continue
			contexts.append(ContextEvidence(chunk=chunk, relation=option.relation))
			seen_text.add(identity)
			seen_selected_ids.add(chunk.chunk_id)
			used_characters += text_size
		expanded.append(replace(hit, context_chunks=tuple(contexts)))
	return tuple(expanded)


def _related_chunks(
	selected: RetrievalChunk,
	document_chunks: tuple[RetrievalChunk, ...],
	nodes_by_id: dict[str, HierarchyNode],
) -> list[_ContextCandidate]:
	selected_ancestors = _ancestor_distances(selected.parent_id, nodes_by_id)
	neighbor_ids = {selected.previous_chunk_id, selected.next_chunk_id} - {None}
	options: list[_ContextCandidate] = []
	for chunk in document_chunks:
		if chunk.chunk_id == selected.chunk_id or chunk.document_id != selected.document_id:
			continue
		parent_distance = selected_ancestors.get(chunk.parent_id)
		if parent_distance is not None:
			options.append(
				_ContextCandidate(
					chunk, "parent", (0, parent_distance, abs(chunk.sequence - selected.sequence))
				)
			)
			continue
		child_distance = _descendant_distance(
			chunk.parent_id, selected.parent_id, nodes_by_id
		)
		if child_distance is not None and child_distance > 0:
			options.append(
				_ContextCandidate(
					chunk, "child", (2, child_distance, abs(chunk.sequence - selected.sequence))
				)
			)
			continue
		if (
			chunk.chunk_id in neighbor_ids
			or selected.chunk_id in {chunk.previous_chunk_id, chunk.next_chunk_id}
		):
			options.append(
				_ContextCandidate(
					chunk, "neighbor", (1, abs(chunk.sequence - selected.sequence), chunk.sequence)
				)
			)
	return sorted(options, key=lambda item: (*item.priority, item.chunk.chunk_id))


def _ancestor_distances(
	node_id: str, nodes_by_id: dict[str, HierarchyNode]
) -> dict[str, int]:
	ancestors: dict[str, int] = {}
	current = nodes_by_id.get(node_id)
	distance = 1
	while current is not None and current.parent_id is not None:
		ancestors[current.parent_id] = distance
		current = nodes_by_id.get(current.parent_id)
		distance += 1
	return ancestors


def _descendant_distance(
	candidate_node_id: str,
	selected_node_id: str,
	nodes_by_id: dict[str, HierarchyNode],
) -> int | None:
	current = nodes_by_id.get(candidate_node_id)
	distance = 0
	while current is not None:
		if current.node_id == selected_node_id:
			return distance
		if current.parent_id is None:
			return None
		current = nodes_by_id.get(current.parent_id)
		distance += 1
	return None


def _text_identity(text: str) -> str:
	return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
