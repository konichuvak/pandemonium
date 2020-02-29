===========
Spinning Up
===========

-----------------------
What is |project_name|?
-----------------------

|project_name| is a library that provides implementations for agents that seek to learn about their environment by
predicting multiple signals from a single stream of experience.

.. _architecture: https://en.wikipedia.org/wiki/Pandemonium_architecture

The name of the project is inspired from the `architecture`_
originally developed by Oliver Selfridge in the late 1950s. His computational model is composed of different groups
of "demons" working independently to process the visual stimulus, hence the name -- |project_name|.

This computational framework that is used in the project is due to Adam White's PhD work on learning predictive
knowledge representations.

The goal of this project is to further develop the computational framework established by Adam White and
express some of the common algorithms in RL in terms of terms of hierarchy of "demons".

------------
Installation
------------

|project_name| requires Python 3.6+ and is available on MacOS and Linux.
To begin experimenting clone the repo and install using pip:

.. code-block:: bash

    git clone https://github.com/konichuvak/pandemonium.git
    pip install pandemonium

.. _virtualenv: https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv

.. note::

    While technically optional, we highly recommend that you use virtualenv_
    to create an isolated python environment in which to install |project_name|.
