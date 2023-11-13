# LLM Side Projects
**N.B. The work showcased here is still a work in progress and is not yet complete**
The overall intention of the project is to develop a very basic Q&A Generative AI model using the HuggingFace🤗 Transformers API with PYTorch as a backend

## Table of Contents
- Project Overview
- Installation
- Component Usage

## Project Overview
Please note that this project has been developed with an educational goal, rather than an attempt to develop an accurate model. The model employs Semantic/Similarity Search (FAISS algorithm) to attempt to answer user prompts (questions) related to creating the content for job ads. It's entirely for education, and the scope of this model is more to showcase how such a model can be developed rather than an attempt to develop an accurate model.

Another note to point out is that even though the code is developed as components (and not notebooks). It does contain several `print()` statements that will display various outputs such as the values in the train/test splits, etc. This is more of a personal preference, since I much prefer working with components than notebooks. Notebook performance (particularly with text data) is notoriously slow and often crashes. Also, developing code in component form does make it much more efficient for moving into production!

## Installation
In order to use the HuggingFace🤗 Transformers API locally, you'll need to create a Virtual Environment. These are the steps:
1. In a command terminal type: `python -m venv venv` (this will create a new virtual environment called 'venv' on your local machine)
2. Activate the Virtual Environment: `venv/Scripts/activate`
3. Install the following dependencies:
```
pip install transformers
pip install 'transformers[torch]'
```
4. If you'd also like to use VS Code (like I do!), please run the following command to start the application using the venv: `code .`

Clone the repo with the following command `git clone https://github.com/VikramsDataScience/LLM_Side_Projects.git` and the sequence of component execution is explained in the following 'Component Usage' section.

## Component Usage
So far, there are only two components that need to be run in the following order:
1. 