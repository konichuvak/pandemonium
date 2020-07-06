===========
Spinning Up
===========

-----------------------
What is |project_name|?
-----------------------

|project_name| is a library that provides implementations for RL agents that seek to learn about their environment by
predicting multiple signals from a single stream of experience.

.. _architecture: https://en.wikipedia.org/wiki/Pandemonium_architecture
.. _Horde: http://incompleteideas.net/papers/horde-aamas-11.pdf
.. _thesis: https://sites.ualberta.ca/~amw8/phd.pdf

The name of the project is inspired from the `architecture`_
originally developed by Oliver Selfridge in the late 1950s. His computational model is composed of different groups
of "demons" working independently to process the visual stimulus, hence the name -- |project_name|.

The pandemonium framework has inspired some of the more recent work such as `Horde`_ by Sutton et. al 2011.
The authors designed a scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction.
Since then, `Horde` was further developed and formalized in Adam White's Doctoral `thesis`_, from which this library
borrows most of the definitions and notation.

The goal of this project is to further develop the computational framework established by the creators of `Horde` and
express some of the common algorithms in RL in terms of hierarchy of "demons".

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
