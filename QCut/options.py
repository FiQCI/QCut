"""User-facing configuration for the cutting workflow.

Options are collected once and carried on :class:`QCut.cutcircuit.CutCircuit`, which
passes them to :class:`QCut.cutcircuit.CutExperiment`. Everything downstream reads them
from there, so adding a knob does not mean widening five signatures.

Most of them have to be fixed before the subcircuits are built, because they decide how
the quasiprobability decomposition is formed and therefore what the coefficients mean. A
:class:`CutOptions` is frozen for that reason. ``deduplicate`` is the exception: it
changes only how an experiment is submitted, so it can be flipped on a built
:class:`QCut.cutcircuit.CutExperiment`. It lives here anyway, so that
:func:`QCut.estimate_run` and :func:`QCut.run_experiments` read one setting and cannot
disagree about what a run costs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from QCut.errors.qcuterror import QCutError

#: How the experiment tensor is built. ``"exact"`` enumerates every combination of QPD
#: terms, ``"sample"`` draws from the quasiprobability distribution, and ``"auto"``
#: enumerates while the exact count stays within ``max_exact_groups``.
ExpansionStrategy = Literal["auto", "exact", "sample"]

#: When to merge runs of gates on the same qubit pair. ``"always"`` merges wherever it
#: lowers the cost of that pair on its own, ``"never"`` leaves every gate alone, and
#: ``"auto"`` costs both whole plans and keeps the cheaper. ``True`` and ``False`` are
#: accepted as ``"always"`` and ``"never"``.
ConsolidateStrategy = Literal["auto", "always", "never"]

#: When to let the two sides of a wire cut exchange the measured outcome. ``"always"``
#: uses it for any block, ``"never"`` for none, and ``"auto"`` only for blocks of at
#: least :data:`MIN_COMMUNICATING_BLOCK` wires, which is where it starts to pay for
#: itself. ``True`` and ``False`` are accepted as ``"always"`` and ``"never"``.
CommunicationStrategy = Literal["auto", "always", "never"]

#: Smallest block ``"auto"`` will use classical communication for.
MIN_COMMUNICATING_BLOCK: int = 2

# What types of cuts the cut finder will consider. ``"both"`` considers wire and gate
# cuts, ``"wire"`` considers only wire cuts, and ``"gate"`` considers only gate cuts.
FinderCutMode = Literal["both", "wire", "gate"]


@dataclass(frozen=True)
class CutOptions:
    """Configuration for a cutting run.

    ``consolidate`` merges runs of gates acting on the same qubit pair into one gate
    before cutting, so the pair costs one cut instead of several. See
    :data:`ConsolidateStrategy`. Merging can cost more than it saves once joint cutting
    is in play, because a merged run of gates about different axes is no longer a
    single-axis rotation and cannot be bundled, so ``"auto"`` compares the two plans
    outright rather than guessing.

    ``joint_rotation_cuts`` cuts parallel two-qubit rotation gates with one joint
    decomposition instead of one each, which costs strictly less in both sampling
    overhead and circuit count.

    ``wire_cut_communication`` lets the two sides of a wire cut exchange the measured
    outcome, which lowers the overhead of a block of ``n`` parallel wires from ``4**n``
    to ``2**(n+1) - 1``. It makes those experiments run in two phases, since the state
    one side prepares depends on what the other measured. See
    :data:`CommunicationStrategy`.

    ``expansion`` decides how the decomposition becomes experiment circuits, see
    :data:`ExpansionStrategy`. Under ``"auto"``, ``max_exact_groups`` is the largest
    exact group count still enumerated rather than sampled. When sampling,
    ``num_samples`` draws are taken, defaulting to ``max_exact_groups``, and ``seed``
    fixes the sampler for a reproducible experiment set.

    ``deduplicate`` submits each distinct experiment circuit once and shares its result
    between every slot it fills, rather than running the same circuit once per group.
    The slot count grows exponentially in the number of pieces while the distinct count
    grows linearly, so this is most of the cost of a run. ``shots`` still means shots
    per circuit, so the total spent drops by the same factor; what it bought before was
    re-measuring circuits already measured. Turn it off to submit one circuit per slot.

    ``finder_candidates`` is how many candidate partitions the cut finder generates and
    costs before keeping the cheapest. METIS returns only the partitioning that
    minimises its own objective, the weighted edge cut, which stops being the true cost
    once cuts share a decomposition, so several are generated and compared on what they
    actually cost. Under a qubit budget they vary by how much imbalance METIS is allowed
    before they vary by seed, since that changes the partition more. Both are fixed,
    which makes the finder reproducible; ``seed`` shifts the seeds as a set.

    ``finder_cut_mode`` limits the finder to wire cuts, to gate cuts, or lets it use
    both. ``finder_num_partitions`` asks for a number of pieces and
    ``finder_max_qubits`` for a size limit per piece, either one number for all of them
    or a list with one entry per partition. With neither given the finder splits in two.
    """

    consolidate: ConsolidateStrategy | bool = "auto"
    joint_rotation_cuts: bool = True
    wire_cut_communication: CommunicationStrategy | bool = "auto"
    expansion: ExpansionStrategy = "auto"
    max_exact_groups: int = 1000
    num_samples: int | None = None
    seed: int | None = None
    deduplicate: bool = True
    finder_candidates: int = 5
    finder_cut_mode: FinderCutMode = "both"
    finder_max_qubits: int | list[int] | None = None
    finder_num_partitions: int | None = None

    def __post_init__(self) -> None:  # noqa: C901
        """Validate the combination."""
        if self.consolidate not in ("auto", "always", "never", True, False):
            raise QCutError(
                f"unknown consolidate strategy '{self.consolidate}', expected one of "
                "'auto', 'always', 'never', True, False"
            )
        if self.wire_cut_communication not in ("auto", "always", "never", True, False):
            raise QCutError(
                f"unknown wire_cut_communication strategy "
                f"'{self.wire_cut_communication}', expected one of 'auto', 'always', "
                "'never', True, False"
            )
        if self.expansion not in ("auto", "exact", "sample"):
            raise QCutError(
                f"unknown expansion strategy '{self.expansion}', expected one of "
                "'auto', 'exact', 'sample'"
            )
        if self.finder_cut_mode not in ("both", "wire", "gate"):
            raise QCutError(
                f"unknown finder_cut_mode '{self.finder_cut_mode}', expected one of "
                "'both', 'wire', 'gate'"
            )
        if not isinstance(self.deduplicate, bool):
            raise QCutError(
                f"deduplicate must be a bool, got {type(self.deduplicate).__name__}"
            )
        if self.max_exact_groups < 1:
            raise QCutError("max_exact_groups must be at least 1")
        if self.num_samples is not None and self.num_samples < 1:
            raise QCutError("num_samples must be at least 1")
        if self.finder_candidates < 1:
            raise QCutError("finder_candidates must be at least 1")
        if self.finder_num_partitions is not None and self.finder_num_partitions < 1:
            raise QCutError("finder_num_partitions must be at least 1")

        # Split on the shape first: a per-partition budget is a list, one shared budget
        # is a number, and the checks for the two have nothing in common.
        if isinstance(self.finder_max_qubits, list):
            if any(q < 1 for q in self.finder_max_qubits):
                raise QCutError("all finder_max_qubits must be at least 1")
            if len(self.finder_max_qubits) <= 1:
                raise QCutError("finder_max_qubits must have at least 2 entries")
            if (
                self.finder_num_partitions is not None
                and len(self.finder_max_qubits) != self.finder_num_partitions
            ):
                raise QCutError(
                    "finder_max_qubits must have the same length as "
                    "finder_num_partitions"
                )
        elif self.finder_max_qubits is not None and self.finder_max_qubits < 1:
            raise QCutError("finder_max_qubits must be at least 1")

    @property
    def consolidate_mode(self) -> str:
        """The consolidation strategy, with ``True`` and ``False`` normalised."""
        if self.consolidate is True:
            return "always"
        if self.consolidate is False:
            return "never"
        return self.consolidate

    @property
    def min_communicating_block(self) -> int:
        """Smallest wire cut block that will use classical communication."""
        if self.wire_cut_communication in ("never", False):
            return 0
        if self.wire_cut_communication in ("always", True):
            return 1
        return MIN_COMMUNICATING_BLOCK

    @property
    def sample_count(self) -> int:
        """Number of draws to use when sampling."""
        return (
            self.num_samples if self.num_samples is not None else self.max_exact_groups
        )

    @property
    def num_partitions(self) -> int:
        """Number of partitions to cut the circuit into."""
        if self.finder_num_partitions is not None:
            return self.finder_num_partitions
        if isinstance(self.finder_max_qubits, list):
            return len(self.finder_max_qubits)
        # One budget for every partition says nothing about how many there are, and
        # neither does having no budget at all, so both split in two.
        return 2

    @property
    def max_qubits(self) -> list[int] | None:
        if isinstance(self.finder_max_qubits, list):
            return self.finder_max_qubits
        if self.finder_max_qubits is not None:
            return [self.finder_max_qubits] * self.num_partitions

    def should_sample(self, exact_groups: int) -> bool:
        """Whether an experiment of ``exact_groups`` combinations should be sampled."""
        if self.expansion == "exact":
            return False
        if self.expansion == "sample":
            return True
        return exact_groups > self.max_exact_groups

    def replace(self, **changes) -> CutOptions:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)


#: Used when no options are passed. Reassign to change the defaults process-wide.
DEFAULT_OPTIONS = CutOptions()


def resolve(options: CutOptions | None) -> CutOptions:
    """Return ``options``, falling back to :data:`DEFAULT_OPTIONS`."""
    if options is None:
        return DEFAULT_OPTIONS
    if not isinstance(options, CutOptions):
        raise QCutError(f"options must be a CutOptions, got {type(options).__name__}")
    return options
