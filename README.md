# MarketingClassification

## Contents

[**1. Background**](#background)

[**2. Motivation**](#motivation)

[**2. Results**](#results)

## <a name="background">Background</a>

[This repository](https://nbviewer.jupyter.org/github/des137/MarketingClassification/blob/master/Model.ipynb) contains models built on [Bank Marketing Data set](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) available from UCI ML repository. The classification goal is to predict wheather a customer will accept the 'CD' (Certificate of Deposit) offer based on various customer related and previous campaign related data.

[This notebook](https://nbviewer.jupyter.org/github/des137/MarketingClassification/blob/master/eda.ipynb) quickly performs the basic data exploration to ascertain the intrgrity of the data.

## <a name="motivation">Motivation</a>
The real motivation to study this particular problem is to show the end-to-end pipeline feature of the **sklearn** library. 
'sklearn' is a remarkably well designed library which let's one quickly prototype a data flow pipeline and test a variety of machine learning models, by chaining a set of *Estimators*, *Transformers*, and *Predictors*. [This notebook](https://nbviewer.jupyter.org/github/des137/MarketingClassification/blob/master/Model.ipynb) demonstrates the applications of the pipeline feature. 10 different models were tested on this particular dataset.

## <a name="results">Results</a>

Among the machine learning models that were tested on this particular dataset, not so surprisingly, Light Gradient Boosting framework produced the best results. Obtained: **Gini = 0.87**, or equivalently, **AUC = 0.93**. 

The metrics/accuracy of the model is equivalent to the [an analysis](https://core.ac.uk/download/pdf/55616194.pdf) performed using CRISP-DM methodology.

