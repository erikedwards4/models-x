# README-dev
This README-dev covers topics for the developer who might want to  
know more about how the repo was made and why.  

The new user only needs to git clone and start running the code.  
This README-dev.md treats several topics in more detail.  

## General Linux installs
If you are on a newish Linux machine, you may need to install some well-known Linux utilities, e.g. (using Ubuntu):  
```
sudo apt-get update
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-driver-590
```
etc.  
Install Linux utils as needed as they arise (Linux gives good messages so it is usually easy to identify what to sudo apt-get install to continue).  

For torch, you'll need nvidia-smi installed (should come with nvidia-driver), and you'll need to know your CUDA version:  
```
nvidia-smi
```
For which I get CUDA Version 13.1, so I use "cu131" below (but modify for other versions).  

## Get started using uv
For general info on uv (uv replaces venv, poetry, etc.), see:  
https://docs.astral.sh/uv  
https://github.com/astral-sh/uv  
https://fmind.medium.com/poetry-was-good-uv-is-better-an-mlops-migration-story-f52bf0c6c703  

General for git:  
```
sudo apt-get -y install gh
git config --global user.name "Erik Edwards"
git config --global user.email "erik.edwards4@gmail.com"
```
May have to do gh auth login or similar.  
  
Install uv and other dependencies:  
```
sudo apt-get -y install python3-pip
pip3 install -U pip
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.14
```

## Make project using uv
Make the project repo using uv:  
```
name="models-x"
descr="Repo for external models in Jax"
uv init --lib --name "${name}" --description "${descr}" "${name}"
cd "${name}"
```
This initializes the git repo with a good default .gitignore, starts a pyproject.toml file, gives an empty README.md file, and starts with a good default src/ layout.  

Now get the new repo to GitHub:  
```
git add .
git commit -m "initial commit after uv init"
org="erikedwards4"
gh repo create "${org}/${name}" --source=. --private --push --description "${descr}"
git push --set-upstream origin master
```
For the last step, you may have to add the .ssh key in the repo in GitHub (Settings > Deploy keys), do gh auth login, etc.  

# Virtual env
Next make the venv (virtual environment), which uv places in the project dir:  
```
uv python pin 3.14
uv venv --python 3.14
source .venv/bin/activate
```
By uv default, the .gitignore will correctly ignore the .venv.  

## API Keys and .env file
API keys may be required for some external packages/APIs.  
Put your own API keys in a .env file within the repo directory.  
Keep the permissions for the .env file at minimum:  
```
chmod 640 .env
```

Make sure the .env is in the .gitignore!
```
echo "" >> .gitignore
echo "# .env file" >> .gitignore
echo "*.env" >> .gitignore
```

## Other .gitignore
Ignore .vscode (if using VSCode or VSCodium):  
```
echo "" >> .gitignore
echo "# VS Code" >> .gitignore
echo ".vscode" >> .gitignore
```

Ignore log file outputs:  
```
echo "" >> .gitignore
echo "# Logs" >> .gitignore
echo "log/" >> .gitignore
echo "*.log" >> .gitignore
```

## Data and model files
Ignore files and subdirs that might be data, model files or other downloaded files.  
These can be huge and/or clutter the repo.  
```
echo "" >> .gitignore
echo "# Data/model files" >> .gitignore
echo "datasets*/" >> .gitignore
echo "*.sqlite3" >> .gitignore
echo "*.bz" >> .gitignore
echo "*.gz" >> .gitignore
echo "*.tar" >> .gitignore
echo "*.zip" >> .gitignore
echo "*.tmp" >> .gitignore
echo "tmp/" >> .gitignore
```

## Repo layout
We'll use the classic src directory layout, as meant for packages or libraries  
or apps; it remains the most standard and makes Python imports the most canonical  
and reliable. This is also initiated by uv when it makes the repo.  

Note that Python imports require underscores rather than hyphens.  
That is why the repo is "models-x" and the src dir is "src/models_x".  

We will have subdirs for major categories of classes that arise:  
```
mkdir -m 764 src/models_x/{text,audio}
```

For Python imports and package best-practices, the init files go herein:  
```
touch src/models_x/__init__.py
touch src/models_x/{text,audio}/__init__.py
chmod -R 764 src
```

## Install JAX with uv
Before installing anything else, it's best to get the hard one over with,  
so that no minor/annoying conflicts (like the exact version of pip) arise.  
From the JAX website, it says to use:  
```
pip install -U "jax[cuda13_pip]"
```
But we are using uv, so we'll use uv add, and Google AI recommends the following with the flag for a recent GPU like the NVIDIA Blackwell:  
```
uv add "jax[cuda13_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
This works!  
Note that this will install reasonably new versions of Numpy, Scipy, opt-einsum and ml-dtypes. It's better to let uv and JAX choose these versions rather than to uv add them first with some other versions. Conflicts otherwise arise easily.  

Also add the better static typing for JAX:  
```
uv add jaxtyping typeguard
```

## Install packages with uv
Now uv add other Python packages with the venv activated:  
```
cd "${repodir}"
source .venv/bin/activate
uv add pydantic pydantic-settings
uv add loguru
uv add transformers
```
One could also uv pip install the packages, but uv add is the higher-level API  
for projects, and it is recommended as it better resolves conflicts and updates  
the project.toml and uv.lock files.  

If we require porting a model from Pytorch to JAX, it is best to install the  
smaller, CPU-only version of torch:  
```
uv add torch --index-url https://download.pytorch.org/whl/cpu
```

## Dev packages and testing
Now uv add packages needed only during development (not during prod or general use).  
These are for checking, testing, debugging and improving Python code in general:  
```
uv add --dev pyflakes pycodestyle flake8 pylint mypy ruff pytest
uv add --dev python-dotenv
uv add --dev types-PyYAML
uv add --dev data-science-types
```
I use these extensively during development. They do:  
- pyflakes -- basic syntax checking  
- pycodestyle -- PEP style checking  
- flake8 -- PEP style checking and a bit of linting  
- pylint -- Python lint checking  
- mypy -- static type checking  
- ruff -- fast Rust-based code from uv, checks code, repo, uv.lock, pyproject.toml, etc.  
- pytest -- most-used unit testing framework for Python  

For pytest, the unit testing code goes as usual in a 'tests' subdir:  
```
cd "${repodir}"
mkdir -pm 764 tests
```

Ignore any stubs made with stubgen:
```
echo "" >> .gitignore
echo "# Stubs" >> .gitignore
echo "tests/stubs/*" >> .gitignore
```

Good to sync, lock and check with:  
```
uv sync --upgrade
uv sync --locked
ruff check
```

## Ephemeral packages
In order to keep the dependencies minimal, do not permanently pip install or uv add any packages that are just being used for a quick test or a quick plot. These can be run like:  
```
uv run --with netron plot_netron.py
```
This is just an example (don't usually use netron)...   
To see your dependency tree, use:  
```
uv tree
```
It can be helpful to look at this before/after uv adds, so you can monitor the growth of the dependency tree.  

## Build backend for uv
Finally, we add a build backend to the pyproject.toml file:  
```
echo "" >> pyproject.toml
echo "[build-system]" >> pyproject.toml
echo "requires = ['hatchling']" >> pyproject.toml
echo "build-backend = 'hatchling.build'" >> pyproject.toml
```
And install the src/models_x directory in the venv in editable mode:  
```
echo "" >> pyproject.toml
echo "[tool.hatch.build.targets.wheel]" >> pyproject.toml
echo "packages = ['src/models_x']" >> pyproject.toml
echo "artifacts = ['src/models_x/py.typed']" >> pyproject.toml
```
And add the py.typed (needed by e.g. mypy):  
```
touch src/models_x/py.typed
chmod 664 src/models_x/py.typed
```
Next uv sync and lock as above.  
Now models_x is like a 3rd-party package that can be used in imports.  
It could even be used outside the repo via a pip install.  
This is a good check that things are minimally working at the repo/package level:  
```
uv build
```

Can also be good to clean:
```
uv cache clean
```

Now we have a src layout with an editable install, which is the industry-standard for a Python ML repo.  
But the immediate purpose was to allow 'from models_x.xxx.xxx import xxx' within the Python code.  

## Questions/comments
erik.edwards4@gmail.com  
