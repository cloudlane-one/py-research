# How to add dependencies

Add a new dependency:

```powershell
poetry add DEPENDENCY_NAME
```

Add a new dependency only for development use:

```powershell
poetry add --group dev DEPENDENCY_NAME
```

This will update the files `pyproject.toml` and `poetry.lock`. Commit and push both files once you have added all dependencies. Other contributors can then pull the changed files and re-run `poetry install` to get the new dependencies themselves.

> While you are at it, it's probably a good idea to also run `deptry .` to check for issues such as ununsed dependencies.

Please check the health and vulnerabilities of new dependency packages before adding them via [Snyk Advisor](https://snyk.io/advisor/python) and [Snyk Security](https://security.snyk.io/) (or [snyk CLI](https://github.com/snyk/cli)). If the project turns out to be in a poor state, consider alternatives.

> **PyCharm GUI Limitation:** Note that even though the [integrated PyCharm GUI for package management](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html) will use Poetry under the hood (after [proper configuration](./how-to-setup-python-env.md)), it lacks the option to add already installed sub-dependency packages (e.g. dependencies of dependencies) as explicitly installed packages to `pyproject.toml`. If you find yourself needing to do that, you will have to resort to `poetry add` in the terminal.
