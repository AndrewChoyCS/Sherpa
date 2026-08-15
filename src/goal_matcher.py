"""Free-text training goal -> target clip in the graph.

The user types "teach the robot to fold a shirt". Something has to decide which clip in
the graph that means, so the path search has a destination.

EgoVerse gives us the language for free: every episode carries ``task_name`` and a
human-written ``task_description`` ("fold the black t shirt using both arms"). Matching a
goal against those strings is a text-retrieval problem over a few hundred short
documents, so this uses TF-IDF cosine similarity rather than a neural sentence encoder --
no model download, no extra dependency (scikit-learn is already required), deterministic,
and instant.

Two vectorisers are combined because they fail in different places:

- **Character n-grams** (``char_wb``, 3-5) survive morphology and compounding. "shirt"
  matches "tshirt" because "shirt" is literally a substring; a word-level model sees two
  unrelated tokens.
- **Word n-grams** (1-2) capture phrase-level intent and keep character-level noise from
  dominating on short queries.

Their cosine similarities are summed, so the combined score lives in ``[0, 2]``.

**This is lexical matching, not semantic.** It has a real, measured failure mode: against
the sample dataset, "tidy up the table" retrieves ``pick_hat`` because both share the
bigram "up the", not because picking up a hat tidies anything. Nothing in TF-IDF can know
that "tidy" relates to "sort utensils". That is why :class:`GoalMatch` always carries the
ranked alternates and an explicit confidence, and why the UI shows them as an override
control instead of silently committing to the top hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Metadata columns concatenated into each clip's searchable document, most specific
# first. task_description is the human-written sentence and carries most of the signal.
DOCUMENT_FIELDS = ("task_name", "task_description", "source", "embodiment")

# A match below this combined score, or with less than this relative lead over the
# runner-up task, is reported as low-confidence. Both thresholds were set by running the
# real task strings: a genuine hit like "fold a shirt" -> yam_fold_tshirt clears them
# comfortably, while the known "tidy up the table" false positive does not.
MIN_SCORE = 0.35
MIN_MARGIN = 0.20

TARGET_SELECTIONS = ("hardest", "medoid", "easiest")

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass
class GoalCandidate:
    """One candidate task family for a goal, with its representative clip."""

    task_name: str
    score: float
    clip_index: int
    episode_id: str
    task_description: str
    n_clips: int

    def label(self) -> str:
        """Human-readable one-liner for a dropdown."""
        text = f"{self.task_name} ({self.n_clips} clips, score {self.score:.2f})"
        if self.task_description:
            text += f" — {self.task_description[:60]}"
        return text


@dataclass
class GoalMatch:
    """The resolved target for a free-text goal, plus enough context to override it.

    Attributes:
        query: The text as typed.
        target_index: Clip index to route the curriculum to.
        task_name: Task family the target belongs to.
        score: Combined char+word cosine similarity, in ``[0, 2]``.
        margin: Relative lead over the runner-up task, in ``[0, 1]``. 1.0 means no
            runner-up scored at all.
        candidates: Ranked task families, best first, for a manual override control.
        is_confident: Whether ``score`` and ``margin`` both clear their thresholds.
        note: Why the match is untrustworthy, when it is. Empty otherwise.
    """

    query: str
    target_index: int
    task_name: str
    score: float
    margin: float
    candidates: List[GoalCandidate] = field(default_factory=list)
    is_confident: bool = False
    note: str = ""


def _normalize(text: object) -> str:
    """Lowercase, split snake_case and camelCase, collapse to spaced words.

    ``yam_fold_tshirt`` and ``foldTshirt`` both become ``yam fold tshirt``, so the
    machine-generated task names match natural phrasing.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    raw = _CAMEL.sub(" ", str(text))
    return _NON_WORD.sub(" ", raw.lower()).strip()


def build_documents(node_frame: pd.DataFrame) -> List[str]:
    """One searchable text document per clip, in dataset order."""
    parts: List[List[str]] = [[] for _ in range(len(node_frame))]
    for column in DOCUMENT_FIELDS:
        if column not in node_frame.columns:
            continue
        for i, value in enumerate(node_frame[column].tolist()):
            normalized = _normalize(value)
            if normalized:
                parts[i].append(normalized)
    return [" ".join(p) for p in parts]


class GoalMatcher:
    """TF-IDF retrieval of a target clip from a plain-English training goal.

    Fitted once per dataset and reused across queries, so the vectorisers are built a
    single time even though the dashboard re-queries on every keystroke-and-submit.
    """

    def __init__(self, node_frame: pd.DataFrame, distance_matrix: Optional[np.ndarray] = None):
        """
        Args:
            node_frame: Per-clip attributes in dataset order, as produced by
                :meth:`~src.pipeline.PipelineResult.frame`. Uses ``task_name``,
                ``task_description``, ``source``, ``embodiment`` and ``difficulty``.
            distance_matrix: Optional ``(N, N)`` DTW distances, needed only for
                ``target_selection="medoid"``.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.nodes = node_frame.reset_index(drop=True)
        self.distance_matrix = distance_matrix
        self.documents = build_documents(self.nodes)
        self._difficulty = self.nodes["difficulty"].astype(float).fillna(1.0).to_numpy()

        self._vectorizers = []
        self._matrices = []
        for kwargs in (
            dict(analyzer="char_wb", ngram_range=(3, 5)),
            dict(analyzer="word", ngram_range=(1, 2)),
        ):
            vectorizer = TfidfVectorizer(sublinear_tf=True, **kwargs)
            try:
                matrix = vectorizer.fit_transform(self.documents)
            except ValueError:
                # Every document empty (no language metadata at all). Retrieval is
                # impossible; scores stay zero and every match reports low confidence.
                continue
            self._vectorizers.append(vectorizer)
            self._matrices.append(matrix)

    # ------------------------------------------------------------------ #
    def clip_scores(self, query: str) -> np.ndarray:
        """Combined similarity of every clip to ``query``, in ``[0, 2]``."""
        from sklearn.metrics.pairwise import cosine_similarity

        n = len(self.nodes)
        text = _normalize(query)
        if not text or not self._vectorizers:
            return np.zeros(n, dtype=np.float64)

        total = np.zeros(n, dtype=np.float64)
        for vectorizer, matrix in zip(self._vectorizers, self._matrices):
            vector = vectorizer.transform([text])
            if vector.nnz == 0:
                # No shared n-grams at all: an out-of-vocabulary query, not a weak match.
                continue
            total += cosine_similarity(vector, matrix).ravel()
        return total

    def task_scores(self, query: str) -> pd.Series:
        """Per-task-family score: the best score among that family's clips.

        Max rather than mean, because one well-phrased clip description is enough to
        identify the family, and averaging would penalise families whose other clips
        have terse descriptions.
        """
        scores = self.clip_scores(query)
        tasks = self.nodes["task_name"].astype(str) if "task_name" in self.nodes else pd.Series(
            ["unknown"] * len(self.nodes)
        )
        return pd.Series(scores).groupby(tasks).max().sort_values(ascending=False)

    # ------------------------------------------------------------------ #
    def _representative(self, clip_indices: np.ndarray, selection: str) -> int:
        """Pick the clip within a task family to aim the curriculum at."""
        if selection == "easiest":
            return int(clip_indices[np.argmin(self._difficulty[clip_indices])])
        if selection == "medoid":
            if self.distance_matrix is not None and clip_indices.size > 1:
                sub = self.distance_matrix[np.ix_(clip_indices, clip_indices)]
                return int(clip_indices[np.argmin(sub.sum(axis=1))])
            return int(clip_indices[0])
        # "hardest": the goal is the capability you want to *end* able to do, so the
        # target is the most demanding clip in the family, not a typical one.
        return int(clip_indices[np.argmax(self._difficulty[clip_indices])])

    def _candidate(self, task_name: str, score: float, selection: str) -> GoalCandidate:
        clips = np.flatnonzero((self.nodes["task_name"].astype(str) == task_name).to_numpy())
        index = self._representative(clips, selection)
        description = ""
        if "task_description" in self.nodes.columns:
            description = str(self.nodes["task_description"].iloc[index] or "")
        return GoalCandidate(
            task_name=task_name,
            score=float(score),
            clip_index=index,
            episode_id=str(self.nodes["episode_id"].iloc[index]),
            task_description=description,
            n_clips=int(clips.size),
        )

    def match(
        self, query: str, top_k: int = 5, target_selection: str = "hardest"
    ) -> GoalMatch:
        """Resolve a free-text goal to a target clip.

        Args:
            query: The training goal in plain English.
            top_k: How many alternate task families to return for manual override.
            target_selection: Which clip inside the winning family to aim at --
                ``"hardest"`` (default), ``"medoid"`` or ``"easiest"``.

        Returns:
            A :class:`GoalMatch`. An empty, gibberish or out-of-vocabulary query does not
            raise: it returns the hardest clip in the dataset as a stand-in target with
            ``is_confident=False`` and an explanatory ``note``, so the UI can prompt for
            a manual pick instead of erroring out.
        """
        if target_selection not in TARGET_SELECTIONS:
            raise ValueError(
                f"target_selection must be one of {TARGET_SELECTIONS}, got {target_selection!r}"
            )
        if self.nodes.empty:
            raise ValueError("cannot match a goal against an empty clip set")

        ranked = self.task_scores(query)
        top = float(ranked.iloc[0]) if len(ranked) else 0.0
        runner_up = float(ranked.iloc[1]) if len(ranked) > 1 else 0.0

        if top <= 0.0:
            # Nothing matched. Fall back to the hardest clip overall and say so.
            index = int(np.argmax(self._difficulty))
            task_name = str(self.nodes["task_name"].iloc[index]) if "task_name" in self.nodes else "unknown"
            note = (
                "No task description shares any wording with this goal. Showing the "
                "hardest clip in the dataset as a placeholder — pick a target manually."
            )
            if not query or not _normalize(query):
                note = "Enter a training goal, or pick a target task manually."
            return GoalMatch(
                query=query,
                target_index=index,
                task_name=task_name,
                score=0.0,
                margin=0.0,
                candidates=[
                    self._candidate(name, float(score), target_selection)
                    for name, score in ranked.head(top_k).items()
                ],
                is_confident=False,
                note=note,
            )

        margin = (top - runner_up) / top
        candidates = [
            self._candidate(str(name), float(score), target_selection)
            for name, score in ranked.head(top_k).items()
            if float(score) > 0.0
        ]
        best = candidates[0]

        note = ""
        if top < MIN_SCORE:
            note = (
                f"Weak lexical match (score {top:.2f} < {MIN_SCORE}). The wording of this "
                "goal barely overlaps any task description — confirm the target below."
            )
        elif margin < MIN_MARGIN:
            note = (
                f"'{best.task_name}' beat '{ranked.index[1]}' by only {margin:.0%}. "
                "These task descriptions are lexically similar; confirm the target below."
            )

        return GoalMatch(
            query=query,
            target_index=best.clip_index,
            task_name=best.task_name,
            score=top,
            margin=float(margin),
            candidates=candidates,
            is_confident=not note,
            note=note,
        )
