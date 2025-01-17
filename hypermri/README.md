# hypermri 🧲

Schilling-AG python module to work with Bruker MRI data.

## Prerequisites
1. GitHub Desktop (or another kind of git managment tool): https://desktop.github.com
2. Anaconda: https://www.anaconda.com/download
   * Should have Jupyter lab installed
    * Make sure to run Anaconda as an Admin so that you can actually install packages
3. Python Editor (e.g. Pycharm Community, downloadable through the JetBrains Toolbox: https://www.jetbrains.com/de-de/lp/toolbox/)


## Installation

The following steps will guide you through the installation process. This process is assuming your are using
GitHub desktop, but is close to identical if you are using git in the terminal.

First we will download the hypermri repository. Click on "Clone" on the repository webpage, select "https".
Open GitHub Desktop, click on "Clone a repository from the Internet", select "URL" and paste the just saved https link in there.
Authenticate with your TUM ID and password.


If you are not using GitHub Desktop but a commandline tool you install  like:
```bash
$ git clone https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t.git
```
Next step we need to create a new python environemt using conda. First go to the path where you have cloned
the repository to:
```bash
$ cd python-code-bruker-7t
# create a new environment
$ conda create -n hypermri_env python==3.10.10
# activate it
$ conda activate hypermri_env
```
As a final step we will install all dependencies into that environment and create a local installation of the hypermri package.
```bash

(hypermri_env)$ pip install -r dev-requirements.txt

(hypermri_env)$ pip install -e ".[dev]"
```
Note : If you encounter any problems check out the **Installation BUGS** section below!

## Usage
```python
# BrukerDir attempts to smartly load in each scan, that is, if a specific sequence-class
# is configured, that specifig class is used to load in the bruker scan.
# Note that every sequence class inherits from the base class BrukerExp. This is also
# the class the smartloader falls back to,  in case it does not recognise the scan.

from hypermri import BrukerDir

# load in the complete experiment directory using the BrukerDir class
# Note: in windows you might have to use r"..." when providing the path

scans = BrukerDir("folderpath/containing/all/scans/from/one/experiment")

# to get an overview over all scans at any time, use:
scans.display()
```

## Installation BUGS

Known bugs:

- [1] `ERROR: Cannot find command 'git' - do you have 'git' installed and in your PATH?`
- [2] `import hypermri` does not work

### [1] `ERROR: Cannot find command 'git' - do you have 'git' installed and in your PATH?`

This happens after running `pip install -r dev-requirements.txt` on windows machines.

You probably don't have git installed in your conda prompt. Try:
``` bash
$ conda install git
```
And now continue where you left of! :)

### [2] `import hypermri` does not work

Steps you can take inside your terminal:
##### 1. Make sure you have the environment activated befor you start a script/notebook:
```bash
   $ conda info | grep active

    > active environment : hypermri_env
    > active env location : .../envs/hypermri_env
```
If this gives you another active environment use `conda activate hypermri`.

##### 2. Have a look if hypermri is install using the package manager *pip*.
```bash
$ pip list | grep hypermri

> hypermri             0.0.1       .../python-code-bruker-7t
```
If nothing is returned here, follow the installation process above.

##### 3. In case, none of the above steps work, try a REINSTALL.
```bash
$ pip uninstall hypermri
$ cd python-code-bruker-7t
$ pip install -e ".[dev]"
```
### [3] Error 404 when opening Jupyter lab
Try uninstalling and reinstalling jupyter lab
``` bash 
pip uninstall jupyterlab
conda install -c conda-forge jupyterlab
```

### [4] Jupyter lab not running in the right harddrive
``` bash
jupyter lab --notebook-dir=D:/
```



## Logging

_*"Don't print, log!"*_

Simply put, logging allows for a more controlled way to print. We can distinguish
between different logging hierachies, let logging behave like the classic print,
log to a file and much more...

The default log-levels/hierachies are:
| Level    | Numeric value |
| -------- | ------------- |
| CRITICAL | 50            |
| ERROR    | 40            |
| WARNING  | 30            |
| INFO     | 20            |
| DEBUG    | 10            |
| NOTSET   | 0             |

Depending on the specified logging level, only messages with an equal or higher level are displayed/output.

##### Simple example: Add a verbose option to a sequence class
```python
from ..utils.logging import LOG_MODES, init_default_logger

# initialize logger
logger = init_default_logger(__name__)

class test():
    def __init__(self, log_mode='critical'):
        
        # globally change log level, default is critical
        logger.setLevel(LOG_MODES[log_mode])

        # log some stuff on different levels
        logger.critical("This is a critical message")
        logger.info("This is a info message")
        logger.debug("This is a debug message")

>> test()
  __main__:This is a critical message
>> test(log_mode='info')
  __main__:This is a critical message
  __main__:This is a info message
```

To log to a file instead of the standard output, you can create your own init function
and simply add another *handler* there:

```python
file_handler = logging.FileHandler('sample.log')
logger.addHandler(file_handler) # this line into the if-statement
```

## Development:

To run pytest (test-cases):
`$ pytest`


### Adding Dependencies

All package handling is done inside the `pyproject.toml` file. Here are the steps necessary to add dependencies to the package:

0. Before adding a dependency to the package, make sure it is compatible! (See **Tox**!)

1. Open the file `pyproject.toml` and go to `[project]`. Then, add the dependency inside the `dependencies`-list. You can use the syntax [listed here](https://python-poetry.org/docs/dependency-specification/) to specify the dependencies.

2. Use the `pip-tools` command to compile dependencies files:
- Install `pip-tools` (contains `pip-compile`) using `pip`:
```bash
$ pip install pip-tools
```

- Generate usage dependencies file:  

```bash
$ pip-compile --output-file=requirements.txt --resolver=backtracking pyproject.toml
```

- Generate development dependencies file:

```bash
$ pip-compile --extra=dev --output-file=dev-requirements.txt --resolver=backtracking pyproject.toml
```

#### Cleaning up the Environment

Sometimes your environemt can get messy after a wild coding session. To synchronise your environment back to the requirements file you can use the `pip-sync` command. This will install/upgrade/uninstall everything necessary to match the *requirements.txt* (or *dev-requirements.txt*) contents. However, you will have to reinstall hypermri (see above on how to install). To clean up your environment, type:

```bash
$ pip-sync requirements.txt` or `$ pip-sync dev-requirements.txt`
```
#### Environment Testing with Tox

Tox is a great tool for environment testing. It creates temporary environments to look if all dependencies are compatible and all testcases are working. Inside the `tox.ini` file you can specify what environemts should be tested. To test the environments specified in `tox.ini`, simply type:

```bash
$ tox
```

Note: This may take a while since tox is actually creating (and afterwards deleting) all specified environments. That is, it will download (and delete again) all given dependencies inside each of the environments in `tox.ini`.


### Use Jupyter notebook extension for Visual Studie Code for easy debugging

For an easier debugging (seeing variables, setting breakepoints), jupyter notebooks using the package can be run in Visual Studie Code.
For this, follow these instructions: https://code.visualstudio.com/docs/datascience/jupyter-notebooks (select your installed Conda environment as a kernel)

In case the ipywidgets throw an error, downgrade the Jupyter extension to v2022.10.110 (see https://github.com/microsoft/vscode-jupyter/issues/14585).