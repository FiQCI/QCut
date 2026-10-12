Options
=======

A cutting run is configured with a :class:`~QCut.options.CutOptions`. Pass one to
``find_cuts``, ``get_locations_and_subcircuits`` or ``run`` and it is carried on the
resulting ``CutCircuit`` and ``CutExperiment``, so it only has to be given once.

.. code:: python

   from QCut import find_cuts, CutOptions

    options = CutOptions(
        finder_num_partitions=3,
        finder_cut_mode="both",
        consolidate=False,
    )

    cut_circuit = find_cuts(circuit , options=options)


Most options are fixed before the subcircuits are built, because they decide how the
quasiprobability decomposition is formed and therefore what its coefficients mean. A
``CutOptions`` is frozen for that reason. ``deduplicate`` is the exception: it changes
only how a built experiment is submitted. Use ``options.replace(...)`` for a modified copy,
or reassign ``QCut.DEFAULT_OPTIONS`` to change the defaults process-wide. For more examples
on using options, see :doc:`QCutOptionsExample.ipynb <notebooks/QCutOptionsExample>`.

Merging gates on the same qubit pair
------------------------------------

``consolidate`` (default ``"auto"``) merges runs of gates acting on the same qubit pair
into one gate before cutting. If nothing else touches those qubits in between, the whole
run is itself one two-qubit unitary, and cutting it once is usually cheaper than cutting
each gate, because the overhead of separate cuts multiplies.

A run does not have to be contiguous. Gates on other qubits commute with it and are
skipped, and a gate that overlaps the pair is moved out of the way whenever it commutes
with the members that would cross it. That last case carries most of the benefit. A
layer of an Ising or QAOA circuit puts a neighbouring ``rzz`` or ``cz`` between every
pair of gates on the same qubits, and treating those as blockers leaves almost nothing
adjacent enough to merge. Which gate is involved matters more than which qubits. A
``cx`` whose *control* lands on the pair commutes with a Z-diagonal run and is moved
past, while the same gate reversed does not and ends the run.

Two ``rzz(0.4)`` gates on a pair cost :math:`\gamma = 3.16` over 36 subexperiments cut
separately, against :math:`\gamma = 2.43` over 6 as the single ``rzz(0.8)`` they compose
to. Two ``cz`` gates compose to the identity, so the merged cut costs
:math:`\gamma = 1`.

Merging is not always the cheaper choice once joint cutting is in play. A run of gates
about *different* axes composes to a generic two-qubit unitary, which is no longer a
single-axis rotation and so can no longer join a joint decomposition. Cutting the gates
separately and bundling each with its parallel partners can beat merging them.

There is no way to tell which wins from one pair alone, so the three strategies are

``"auto"``
    Split the circuit both ways, cost each plan in full including the bundles it allows,
    and keep the cheaper. The comparison is exact rather than a heuristic, and it is
    optimal in every case.

``"always"`` (or ``True``)
    Merge wherever it lowers the cost of that pair considered on its own.

``"never"`` (or ``False``)
    Leave every gate alone.

Under ``"auto"``, ``find_cuts`` also runs the whole cut search twice, because consolidating
changes which edges the partitioner sees and the two plans can end up cutting different
gates entirely. Both paths log which way they went and what it saved at INFO level.

A run holding only one two-qubit gate is left alone, since absorbing the surrounding
single-qubit gates cannot change :math:`\gamma` and would only replace a named gate to a
generic unitary.

Note that merging can turn two separately placed cut markers on one pair into a single
cut, which changes ``len(cut_locations)``. That is intended. QCut already requires
(since otherwise partitioning would not be possible) that if one gate on a pair is cut
then every gate on that pair is cut, so the gates in a run all have to be cut anyway.

The pass is also available on its own as
:func:`~QCut.consolidate.consolidate_two_qubit_blocks` with some extra control knobs,
for running on a circuit before placing cuts by hand.

Cutting parallel rotation gates together
----------------------------------------

``joint_rotation_cuts`` (default ``True``) cuts several parallel two-qubit rotation gates
with one decomposition instead of one each. Separate cuts cost the product of their
overheads, which is not optimal for gates equivalent to a rotation about a single Cartan
axis. Cutting :math:`n` of them together costs
:math:`2\prod_s (1 + |\sin\theta_s|) - 1` against
:math:`\prod_s (1 + 2|\sin\theta_s|)`.

Two CNOT gates therefore cost :math:`\gamma = 7` over 30 subexperiments rather than
:math:`\gamma = 9` over 36, and three cost :math:`\gamma = 15` over 132 rather than
:math:`\gamma = 27` over 216. Shot count goes as :math:`\gamma^2`, so three parallel
gates get about three times cheaper and four about seven times. Both the overhead and the
circuit count fall, so there is nothing to trade off, which is why it defaults to on.

The gates have to be equivalent to a single-axis rotation, which covers ``rzz``, ``rxx``,
``ryy``, ``rzx``, the controlled rotations, ``cp``, ``cx``, ``cz`` and ``ecr`` but not
``swap``, ``iswap``, ``dcx`` or ``xx_plus_yy``. They also have to be parallel and to join
the same pair of subcircuits, because each side of the decomposition acts on several
qubits of one subcircuit at once. Anything that does not qualify is cut on its own as
before, so turning this on can only help. ``QCut.bundle.plan_bundles`` reports what it
grouped and what that saved at INFO level.

Wire cuts with one-way classical communication
----------------------------------------------

``wire_cut_communication`` (default ``"auto"``) lets the two sides of a wire cut exchange
the measured outcome. Cutting :math:`n` wires locally costs :math:`4^n` and that is
provably the best possible, so a block of wires gains nothing on its own. Communicating
brings it down to :math:`2^{n+1} - 1`, and the number of circuit groups/pairs from :math:`8^n` to
:math:`2^n(2^{n+1}-1)`.

.. list-table::
   :header-rows: 1

   * - wires
     - communicating
     - local only
   * - 1
     - 6 circuits, :math:`\gamma = 3`
     - 8 circuits, :math:`\gamma = 4`
   * - 2
     - 28 circuits, :math:`\gamma = 7`
     - 64 circuits, :math:`\gamma = 16`
   * - 3
     - 120 circuits, :math:`\gamma = 15`
     - 512 circuits, :math:`\gamma = 64`

The circuit count is the reliable win. The shot cost is more subtle, because
:math:`\gamma` assumes the prepared state can follow the measured outcome shot by shot.
Batched runs emulate that by post-selection, which reads a group's measuring side from
only the shots whose label matched and rescales by :math:`2^n`. Sharing that circuit
between the channel's groups pays for the rescaling exactly, so the measuring side comes
out at parity rather than ahead, and the realised gain falls short of the ratio of the
:math:`\gamma`\ s. How far short depends on the circuit. Measurements put a single wire
behind the local tables on shots and blocks of two and more ahead on shots, circuits and
jobs together.

So the three strategies are

``"auto"``
    Communicate only for blocks of at least
    :data:`~QCut.options.MIN_COMMUNICATING_BLOCK` wires, two by default, which is where
    it starts to pay for itself. A single wire trades less shots for more circuits.

``"always"`` (or ``True``)
    Communicate wherever the cuts allow it, single wires included.

``"never"`` (or ``False``)
    Keep the local decomposition everywhere.

Communicating makes those experiments run in **waves**. Wave one holds everything no
measured label can affect, shared between the groups that differ only in which label they
answer, and each later wave has its shots split in proportion to how often the labels it
depends on came up. There are as many waves as the dependencies are deep, so a circuit
split into A, B and C where A feeds B and B feeds C takes three, because B's own measured
outcome is what decides what C prepares.

The waves do not get equal shares of the budget. A measuring shot buys precision for
every group answering its shared circuit at once, while a preparing shot buys it for one
group, so an even split over the subcircuits over-funds the measuring wave. It takes
:data:`~QCut.circuit_knitting.MEASURE_SHARE` of the run instead, a sixth, and the later
waves divide the rest evenly. The total is the same either way, so ``shots`` means what
it always did.

Nothing runs one shot at a time, and batching is preserved. Within a wave the circuits are
sorted by how many shots they want and grouped into batches, each running at the mean of
what its own circuits asked for. A batch closes when it reaches ``max_batch_size`` or when
its requests span more than a factor of two, whichever comes first. The first limit keeps
a job within what the backend takes, and the second stops a generous ``max_batch_size``
from putting a whole wave in one job at one shot count, which would be uniform allocation
and would throw the proportional split away. So a wave costs a handful of jobs whatever
the batch size is. ``CutExperiment.communicates`` says whether an experiment took that
path.

A communicating cut forces its measuring side to run before its preparing side. If wire
cuts point both ways between the same two subcircuits there is no valid order, so one
direction keeps the local decomposition and says so at INFO level. The derivation is on
the :doc:`classical communication <theory/LOCC_wire_derivation>` page.

Sampling instead of enumerating
-------------------------------

Building every combination of QPD terms costs the product of the per-cut term counts. That
is fine for a relatively small number of cuts and intractably expensive beyond that,
since a generic two-qubit gate needs 58 terms. ``expansion`` controls what happens instead.

``"auto"`` (the default) enumerates while the exact count is within ``max_exact_groups``
(default 1000) and samples above it. ``"exact"`` always enumerates, ``"sample"`` always
samples.

When sampling, terms are drawn per cut with probability proportional to :math:`|c|`, and
``num_samples`` draws are taken (defaulting to ``max_exact_groups``). Repeated draws are
collapsed, so drawing 500 times from a six term decomposition still builds six circuits.
Set ``seed`` for a reproducible experiment set.

Sampling is not only a way to make large cases tractable. Enumerating gives every group the
same number of shots and corrects with a weight afterwards, which is higher variance than
drawing proportional to :math:`|c|` in the first place. The estimator is unbiased either
way, and ``CutExperiment.sampled`` says which path was taken.

Submitting each distinct circuit once
------------------------------------

``deduplicate`` (default ``True``) submits each structurally distinct experiment circuit
once and shares its result with every slot it fills.

A subcircuit's experiment circuit depends only on the QPD terms of the cuts touching that
subcircuit and on the measurement setting over its own qubits. Neither changes between
most of an experiment's groups, so the same circuit is built over and over: the number of
slots grows exponentially in the number of pieces while the number of distinct circuits
grows linearly. A three-piece chain with two gate cuts and five observables holds 324
circuits and 45 distinct ones; a four-piece one holds 2592 and 70.

``shots`` still means shots per circuit, so a run spends that many on each distinct
circuit and the total drops by the same factor. What the old total bought was
re-measuring circuits already measured. Sharing a sample between groups leaves the
estimate unbiased, because a group's subcircuits are multiplied together while the groups
themselves are summed, and no group reads one circuit twice. Measured over 15 seeds on
the three-piece chain, the deduplicated estimator's error was *smaller* than the
undeduplicated one's at the same shots per circuit, and six times smaller at an equal
total shot budget, because the quasiprobability coefficients come in signed pairs whose
variance contributions largely cancel.

``CutExperiment.num_circuits`` still counts the circuits the decomposition calls for.
``estimate_run()`` counts what is actually submitted. Experiments whose wire cuts
communicate are unaffected; their first wave already shares circuits between the groups
that differ only in the label they answer.

Configuring cut finding options
-------------------------------

``CutOptions`` also carries the options that ``find_cuts`` uses to steer the partitioner.
These are documented on the :doc:`automatic cut finding <examples/AutomaticCuts>` page, and the defaults are
``CutOptions.finder_num_partitions = 2``, ``CutOptions.finder_cut_mode = "wire"``,
``CutOptions.finder_candidates = 5``, ``CutOptions.finder_max_qubits = None``, and
``CutOptions.finder_seed = 0``.

``finder_num_partitions`` and ``finder_max_qubits`` control the amount and size of the partitions,
``finder_cut_mode`` controls whether to place wire cuts, gate cuts or both, and
``finder_candidates`` controls how many different partitionings to test before picking the cheapest.
The seeds are fixed, so the finder is deterministic and reproducible.
