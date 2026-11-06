<h1>Contributing to QCut</h1>

<h2>Create a fork of QCut</h2>

In GitHub navigate to the QCut repository and click on fork. Once you have your fork of QCut clone it to your local machine.

<h2>Prepare environment</h2>

Using uv initialise the development environment:
```shell
uv sync
```

<h2>Working on you changes</h2>

Create a new branch for your changes. Once you are done lint and type check all files to make sure you are following the proper style:

```shell
uv run ruff check --fix
uvx ty check QCut
uv run codespell
```

after running ruff and ty fix any remaining issues. A real word codespell flags belongs
in `ignore-words-list` in `pyproject.toml`.

Also make sure that all tests pass:

```shell
uv run pytest --cov
```

Wall clock in this suite is almost entirely simulated shots: a cut experiment enumerates
one circuit per combination of QPD terms, so the count multiplies with every cut, and
each circuit then wants thousands of shots. Qiskit already spreads a single simulation
across every core, so running the tests in parallel does not help. Three tiers are
marked for that reason:

```shell
uv run pytest -m "not sim"     # ~5s, no simulation at all: QPD tables, gammas, planning
uv run pytest -m "not slow"    # ~95s, everything but the heaviest end-to-end cases
uv run pytest                  # ~230s, the lot
```

Use the first while working, the last before pushing. New tests join the quick tier
automatically. Add `@pytest.mark.sim` to anything that puts a circuit on a simulator,
and `@pytest.mark.slow` as well if it costs more than about ten seconds.

The pytest tests will be ran against multiple Python versions using Github actions when you make a pull request.

If your changes should retain backwards compatibility to older Qiskit versions additionally run:

```shell
uv run tox --parallel
```

Finally make sure that documentation is up to date and builds with no warnings, which CI
requires:

```shell
uv sync --group docs
cd docs
uv run sphinx-build -b html . build/sphinx/html -W -E
```

To run everything CI checks except the tests, which is ruff, ty and the docs build, in one
go:

```shell
uv run tox -e ci
```
