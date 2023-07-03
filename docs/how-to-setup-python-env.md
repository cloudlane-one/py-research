# How to setup a local Python environment

This guide will walk you through how to set up everything you need on your local machine to be able to run code in this repo.

## 1. Install the system requirements

Certain tools have to be installed directly on your workstation system. These are:

- [Git](https://git-scm.com/)
- [Python](https://www.python.org/) (see `pyproject.toml` for current version)
- [Poetry](https://python-poetry.org/)
- Editor with rich Python support (= Python IDE, one of these):
  - [VSCode](https://code.visualstudio.com/)
  - [PyCharm](https://www.jetbrains.com/pycharm/)
- [Pandoc](https://pandoc.org/installing.html) (If you want to generate HTML or PDF docs pages)

> In the Python install wizard, select to **add Python to the Path**, if asked.

Poetry will ask you in its output to add something to your `PATH` environment variable. If you don't know how to do this, please follow [this guide](https://www.c-sharpcorner.com/article/how-to-addedit-path-environment-variable-in-windows-11/). Be sure to edit the `User variables` and **not** the `System variables`, if you do not have admin privileges.

If you have full admin privileges on your system, you should be able to install all tools via their official installation instructions in the order they are listed above. In case you do not have admin privileges see below.

### Special instructions for installation without admin privileges

- Python: make sure to select **not** to install Python for other users
- VSCode: download the **User Installer** package

## 2. Optional: Learn Git basics

If you are new to version control via Git, you might want to take a quick detour into [Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials/what-is-version-control) to get an understanding of the basics.

> The easiest authentication method for Git hosts with Single-Sign-On are [HTTP access tokens](https://confluence.atlassian.com/bitbucketserver/personal-access-tokens-939515499.html). You'll likely want to configure git to store them for you locally via `git config --global credential.helper 'store'`.

## 3. Clone the repo and and set it up

After you installed all tools, [clone this repo](./how-to-clone-this-repo.md) to a directory of your choice and open this directory in your IDE. The rest of this section will be different depending on which IDE you use.

### Option A: Setup repo in VSCode

Open an [integrated terminal](https://code.visualstudio.com/docs/terminal/basics) (PowerShell on Windows) and run `poetry lock & poetry install` to install all Python dependencies.

You will then have to tell VSCode to use the Python environment, which `poetry install` created during the previous step, by [selecting the appropriate Python Interpreter](https://code.visualstudio.com/docs/python/python-tutorial#_select-a-python-interpreter). You can find out how this environment is called and where it is located on your machine via `poetry env info`. Note that VSCode might require a restart before it displays the new environment for selection in the GUI.

When opening this repo for the first time, VSCode should ask you in a popup whether you want to install the extensions recommended by the workspace. Please do so. If you the popup doesn't show, see `/.vscode/extensions.json`) for a list of all extension ids. You can use them as search terms in the Extensions tab.

### Option B: Setup repo in PyCharm

You may need to manually [configure PyCharm to use Poetry](https://www.jetbrains.com/help/pycharm/poetry.html) for dependency management, if it doesn't do so automatically upon opening the repo folder. You can find out where the root Poetry environment is located on your machine via `poetry env info` in the terminal.

Once PyCharm knows you want to use Poetry for dependency management, it should ask you in a Popup whether you want to install or update dependencies (on first setup and whenever dependencies change). Please do so.

> In case installation via the GUI doesn't work, open an integrated terminal (PowerShell on Windows) and run `poetry install`.

This repo configures `black` to auto-format every file on save. For this to work, please [activate the File Watchers plugin](https://www.jetbrains.com/help/pycharm/using-file-watchers.html#ws_file_watchers_bedore_you_start) in PyCharm.

## Final Remarks

To run CLI tools installed by poetry such as `sphinx` or `pytest`, you first need to activate the repo's virtual env for the open terminal. Your IDE should do this automatically for newly opened terminals after you've selected the virtual env in the GUI. If this isn't the case, you can either run `poetry shell` to switch the terminal's Python environment permanently or use `poetry run COMMAND` to run a specific command from within the virtual env.
