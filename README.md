# Literary-predictor

This project is the implementation of my coursework on the topic: *Predicting the user's evaluation of a literary work based on lexical and syntactic parameters*


## Abstract

Recently, finding a correlation between the structural features of the text and its reception has become a more and more widespread task of computational linguistics. However, the interrelation of the reader's perception of a literary work suffers from the ambiguity of many textual parameters by ones it is possible to establish a dependence with a user assessment. Concerning Russian-language literature, such a process is complicated by the lack of representative databases of user reviews and rather noticeable discrepancies with the methods of text data analysis. In this paper, I propose a task of investigating the possibility of predicting the rating of a text only by its lexical and syntactic features.

There are five steps to execute:
- Collect data and build a corpus
- Parse LitRes to scrape text's evaluation
- Extract lexical and syntactic features
- Determine if there is any correlation between the rating and the complexity of the text
- Build a model to predict the estimate


## Corpus
Literature was taken mostly from Moshkov Library and other sources with books in the public domain.

## Rating parser
“LitRes” parser is designed, literary ratings from “LitRes” and “LiveLib” are scraped.
 
 ## Features extraction
Extractor is designed. Various TTR, readability score and others parameters were gain.


## Prediction 
RandomForest, CatBoost, Logistic and Linear regression, KNeighbors models were compared. Since R^2 for the regression problem often took a negative value, it was decided to move on to the classification problem. The papers were divided according to the median of their grades into two classes – "successful" and "unsuccessful". The results of the predictions are presented below.

| Model                     | ROC AUC score|
| -----------               | -----------  |
| RandomForestClassifier    | 0.614        |
| CatBoostClassifier        | 0.614        |
| LogisticRegression        | 0.632        |
| KNeighborsClassifier      | 0.592        |


# Results
Finally, all the goals were achieved. Despite the fact that there is rather weak correlation between the rating and the parameters, it was possible to build models with poorly acceptable quality to prove the hypothesis.



