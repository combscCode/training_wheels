# Training Wheels

Putting guardrails around sklearn.

Training Wheels is a package designed to help students avoid typical beginner's
mistakes when learning sklearn. In a complex machine learning library there
are an infinite number of ways to shoot yourself in the foot. Training Wheels
provides tools that helps you learn about models safely, programmatically
warning you if you're broken any assumptions the model relies on.

## Set up

Create a python 3.9 virtual environment with your preferred software. We recommend
using conda.

```
conda create -n training_wheels python=3.9
conda activate training_wheels
```


## Building locally

Now that your virtual env is working, you'll need to install dependencies. To download
training_wheel's dependencies, run `pip install -r dev_requirements.txt`. Then, to
install `training_wheels` itself, run  `pip install -e .` in the same directory that 
`setup.py` lives in.

## Running tests

We use pytest as our testing framework. Run `make tests` to run our test suite. Tests
should be run anytime a branch is merged with `main`.


hello world