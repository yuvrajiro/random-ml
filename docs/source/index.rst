Welcome to Random-ML's Documentation!
======================================

**`random-ml`** is a machine learning package focused on **randomized neural networks** and **functional link architectures**.
It includes **Random Vector Functional Link (RVFL) networks**, ensemble learning techniques, and various randomized machine learning models.


What is Random-ML?
==================
Traditional neural networks rely on **gradient-based learning** for optimization.
However, randomized networks like **RVFL** use **fixed random weights** for hidden layers, allowing:
- **Faster Training** (no backpropagation)
- **Closed-Form Solutions** (Ridge Regression, Moore-Penrose Pseudoinverse)
- **Good Generalization Performance**



.. grid:: 2
    :gutter: 3
    :class-container: overview-grid

    .. grid-item-card:: Install :fas:`download`
        :link: install
        :link-type: doc

        The easiest way to install random-ml is to use
        PyPI by running::

          pip install random-ml

        Alternatively, github can be forked and cloned to install the package.


    .. grid-item-card:: Reference :fas:`book-open`
        :link: user_guide/index
        :link-type: doc

        The work is a part of MLPedRVFL paper, The preprint is submitted to Pattern Recognition Letter


    .. grid-item-card:: API Reference :fas:`cogs`
        :link: api/index
        :link-type: doc

        The reference guide contains a detailed description of the scikit-survival API. It describes which classes and functions are available
        and what their parameters are.


    .. grid-item-card:: Contributing :fas:`code`
        :link: contributing
        :link-type: doc

        The package is in an early phase of development and we welcome contributions from the community. This guide explains how to contribute to the project.



Key Features
------------
- ✅ **RVFL Variants** (Basic RVFL, Extended RVFL, and MLPedRVFL)
- ✅ **Ensemble Learning** → Bagging & Boosting for RVFL
- ✅ **Flexible Activation Functions** (`relu`, `sigmoid`, `tanh`, etc.)
- ✅ **Direct Link Option** (For RVFL models)


Future Plans
------------

- 📈 **More Randomized Models** (ELM, SCN, etc.)
- 📈 **Deep Randomized Networks** (Randomized Deep Learning)
- 📈 **Optimization Techniques** (Particle Swarm Optimization, Genetic Algorithms)
- 📈 **Model Interpretability** (Feature Importance, SHAP Values)
- 📈 **Model Selection** (Cross-Validation, Hyperparameter Tuning)
- 📈 **Model Evaluation** (Metrics, Plots, etc.)
- 📈 **Model Deployment** (Serialization, Web APIs, etc.)




.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   user_guide/index
   api/index
   Contribute <contributing>
   release_notes
   install
   Cite <cite>





