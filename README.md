# LLM Side Projects
The overall intention of the project is to develop a proof of concept Q&A Generative AI model using the Hugging Face🤗 Transformers API with PYTorch as a backend.

## Table of Contents
- Project Overview
- Installation
- Component Usage

## Project Overview
Please note that this project has been developed with an educational goal, rather than an attempt to develop an accurate model. The model employs Semantic/Similarity Search (scaled Dot Product) to attempt to answer user prompts (questions) related to displaying the content for job ads. It's entirely for education, and the scope of this model is more to showcase how such a model can be developed rather than an attempt to develop an accurate model.

Another note to point out is the code is developed as components (and not notebooks). This is more of a personal preference, since I much prefer working with components than notebooks. Notebook performance with text data is notoriously slow and often crashes. Also, developing code in component form does make it much more efficient for moving into production!

On the note of production, if I have time, it's my hope to be able to develop a basic ML Pipeline using my local machine and Apache Airflow as a possible MLOps stack, since Airflow does allow a free local `pip` installation. Otherwise, in a real life scenario I would use host everything on the cloud with AzureML or SageMaker.

## Installation
In order to use the Hugging Face🤗 Transformers API locally, you'll need to create a Virtual Environment. These are the steps:
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
**N.B. Please contact me for the job ad corpus data. I haven't publicly shared the Seek job ads data upon which the model was built.**

So far, there are only two components that need to be run in the following order:
1. **'Job_Ad_Q&A_PreProcessing.py':** 
    - Change folder into `cd .\src_preprocessing\`
    - Run the component by typing `python '.\Job_Ad_Q&A_PreProcessing.py'`
    - Once this has successfully run 
2. **'Job_Ad_Q&A_Train_Test_Tokenize.py':**
    - Change folder into `cd .\src_train_test_tokenize\`
    - Run the component by typing `python '.\Job_Ad_Q&A_Train_Test_Tokenize.py'`
