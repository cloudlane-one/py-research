# Coding Conventions and Tools

There are many aspects to be considered and architectural decisions to be made for every professional software development project. To avoid wasting time on re-inventing the wheel and get started with implementing the project's actual goals right away, one should rely as much as possible on existing standards, frameworks and packages. This document lists the ones chosen for this project.

## Dependency Definition and Management

Dependencies for this repo have to be listed in `pyproject.toml` (as per the community standard [PEP 621](https://peps.python.org/pep-0621/)) and ideally installed in a dedicated virtual environment. The tool [poetry](https://github.com/python-poetry/poetry) combines all of that.
See the [how-to-guide](./how-to-add-dependencies.md) for more infos.

Health and vulnerabilities of new dependency packages should be investigated via [Snyk Advisor](https://snyk.io/advisor/python) and [Snyk Security](https://security.snyk.io/).

## Data Format & Processing

For processing data, especially in large quantities, there exist well-defined formats, packages and workflows within the Python ecosystem. Please try to rely on these standard methods wherever applicable:

- Perform mathematical operations on homogeneous data in [vectorized form](https://www.askpython.com/python-modules/numpy/vectorization-numpy) via [Numpy](https://numpy.org/) arrays.
- Represent heterogeneous data with [Pandas dataframes](https://realpython.com/pandas-dataframe/).
- If nested data (e.g. dicts) has to follow a certain schema, represent that schema with [Python dataclasses](https://realpython.com/python-data-classes/) or [Pydantic models](https://pypi.org/project/pydantic/).
- If you process large-scale data (> 1 GB), where operations may be slow and objects might not fit into RAM on all machines, try to parallelize your functions with [Ray](https://pypi.org/project/ray/) and/or [Dask](https://pypi.org/project/dask/).

## Programming Paradigms

Try to use a [Functional Programming (FP)](https://en.wikipedia.org/wiki/Functional_programming) style wherever this is feasible without too much effort. The mental model of programming which comes with FP is very close in nature to mathematical thinking, which helps to bridge gaps and ease transfer from theory to application in a data-scientifc context.

In practice, this means that you should keep your functions small and [pure](https://en.wikipedia.org/wiki/Pure_function), rely on vectorized functions rather than loops over single values, and try to avoid objects with internal [mutable state](https://softwareengineering.stackexchange.com/questions/235558/what-is-state-mutable-state-and-immutable-state) as much as possible.
Please see this [tutorial](https://docs.python.org/3/howto/functional.html) to find out what this means for Python code in particular.

## Logging and Status Messages

If you have long-running tasks inside your script or notebook, please try to use the package [tqdm](https://pypi.org/project/tqdm/)

To other, custom status messages while code is running, be it for progress updates, warnings, errors or else, do not use plain `print()` statements. Instead rely on the [structlog](https://pypi.org/project/structlog/) package, which enables you to send out log messages in a well-defined format. These messages can then be picked up by existing logging or status reporting pipelines. This style of logging is called Structured Logging (see [here](https://reflectoring.io/structured-logging/) for one explanation).

For both of these use cases, you can find some convenience functions in the module {py:mod}`py_research.telemetry`.

## Type Hints and Linting

Please use [Type Hints](https://realpython.com/python-type-checking/) in your code (especially your function parameters). These are specified as a standard tool for type-checking in Python back in 2015 via [PEP 484](https://peps.python.org/pep-0484/) and using them has come to be an expectation for modern Python code.

This comes with some overhead, as type hints can be hard to understand at first for people new to coding, yet we still strongly encourage you to use them, as they make your functions easier to understand, document, test and just overall reduce errors.

You can do many things based on these annotations, for instance you can have IDEs pick up the type hints and make according suggestions to the developers who use your API, use them to generate function signatures for hosted documentation, generate sample data for test cases based on them, or build a [CLI](https://typer.tiangolo.com/) or [REST-API](https://fastapi.tiangolo.com/) around them.

These tools are to be used to check code in `/src` before committing it:

- [ruff](https://pypi.org/project/ruff/)
- [pyright](https://pypi.org/project/pyright/) (Fallback for non-VSCode editors: [mypy](https://pypi.org/project/mypy/))

IDEs should be configured to run linting and type-checking live as you edit.

## Code Formatting

Please us the [black](https://pypi.org/project/black/) auto-formatter to enforce the Black code style on all code in `/src` and `/tests`. This serves two important purposes: (1) Reduce mental overhead by not having to think about code style anymore, and (2) Make sure all code in this repo looks similar and is easily readable by all contributors.

IDEs should be configured to run code-formatting on every file save.

## Documentation

Documentation can and should serve mutiple purposes, which generally fall into these four categories:

- Explanations
- Tutorials
- How-To Guides
- References

See [Diátaxis Framework](https://diataxis.fr/) for explanations.

For references, specifically this repo's API references, please write docstrings within your Python code. These will be extracted when performing a documentation build. For docstring format, please follow the [Google Docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html) standard with type info contained in Python type hints.

The other types of docs, which are more prose in nature, have to be created manually in `/docs` and can be either Markdown files or Jupyter notebooks.

Generation of this repo's documentation relies on [Sphinx](https://www.sphinx-doc.org/en/master/). See the [how-to guide](./how-to-generate-docs.md) for more infos.

## Tests

Try to cover as much of the code in `/src` as possible with [unit tests](https://en.wikipedia.org/wiki/Unit_testing). Unit tests are realized in this repo via the [pytest](https://pypi.org/project/pytest/) framework.

Ideally, you should create tests in `/tests` along with any new code in `/src` (or even in advance). This ensures your code is working as expected and also tends to help with design choices. Python files containing tests have to start with `test_`.

For complex test cases, [hypothesis](https://pypi.org/project/hypothesis/) is a good tool to generate sample data. Test coverage should be analyzed via [pytest-coverage](https://pypi.org/project/pytest-cov/) (relying on [coverage.py](https://pypi.org/project/coverage/)).

All tests in `/tests` have be executed and passed before merging into `main`, ideally they are executed even before making a pull request for the merge.

## Commits

A commit should have **one specific type** as listed below. The message you associate with your commit should mention this type and deliver a concise description of the introduced changes. Also, it should have a minimum level of structure, so that automation of tests and releases is easier to implement down the line. If necessary, use the tools of your IDE to split your changes into multiple commits.

In particular, please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) community standard and format your messages like so:

```
<type>[(scope)]: Subject

[Body]
```

Scope and body are optional. Type can be **one of**:

- `feat`: Code changes in `/src` which introduce new feature(s) (May include associated docs).
- `fix`: Code changes in `/src` which fix a bug.
- `perf`: Code changes in `/src` or build methods, which improve performance.
- `refactor`: Other code changes in `/src`.
- `docs`: Only changes in documentation.
- `tests`: Only changes in tests.
- `exp`: Only changes within `/exp` folder.
- `chore`: Other changes (for repo maintenance).

If you write a body, please add issues references at the end.

## Changing these choices

A considerable amount of work was put into researching the conventions and dependencies presented here, especially for ensuring that all of them can work together in unison, so this setup should be a solid starting point for productive Python development. Still, it's only meant as a starting point. There can't be one formula that fits all applications, so as this repo is further developed, copied or forked, this guide should evolve alongside. Also, many of the architecture and workflow decisions were close calls between multiple alternatives. As long as things stay consistent and compatible, there's no reason not to change certain components when deriving new repos from this one. Just search the internet for "alternatives to X" with X being any of the choices of this guide, e.g. "alternatives to pytest".
