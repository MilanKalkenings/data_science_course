# data science roadmap
roadmap on learning data science basics from scratch partitioned into 14 chapters:

1-3: analysing small scale data sets with python on your own
4-5: machine learning for tabular data
6-8: analysing bigger data sets, usually in group settings
9-14: machine learning for non-tabular data

## 1: python basics
learn the basics of python programming

- int, float, char, str
- bool, comparisons, conditionals (if, elif, else)
- list, dict
- loops (for)
- functions (function heads, return)
- classes (constructor, self, getters, setters, scopes, polymorphism)
- import
- exceptions

## 2. data handling basics (pandas)
clean data and engineer features for analysis with pandas 

- data frames (creation, shape, head)
- dropping (drop rows, drop columns)
- indices (change column ids, change row ids)
- indexing and slicing (columns and rows)
- masking (return rows that fulfill certain conditions)
- null values (isna, dropna, fillna)
- creating new columns form existing ones (apply)
- merging data frames (inner, outer, left, right)
- sorting (1 priority, 2 priorities)
- uniques (unique, nunique, value_counts)
- aggregations (max, idmax)
- groupby

## 3. exploratory data analysis (pandas, matplotlib, scipy)
means of statistics to gain insights into your data and to test your assumptions

- summary statistics (mean, median, std, var)
- probability distributions
- hypothesis testing
- boxplot
- histplot
- countplot
- violinplot
- heatmap
- scatterplot
- lineplot

## 4. machine learning basics (sklearn)
machine learning to mine patterns in your data and to solve a variety of problems

- train / val / test split
- linear regression
- logistic regression
- decision tree
- kmeans
- hyperparameter optimization (coarse to fine grid search)

## 5. machine learning best practices
achieve better results

- feature engineering (linearity, monotony, correlation, collinearity)
- error decorrelation
- ensembling (random forest, xgboost)
- feature selection

## 6. pycharm and git
share your code with others and benefit from version control

- pycharm and project structure
- create github repository
- commit
- push
- pull

## 7. sql data bases
sql data bases in python to handle data sets

## 8. mongodb
non-sql data bases in python to taim "big data"

## 9. deep learning basics with torch
torch for deep learning projects

- tensors
- loss functions
- stochastic gradient descent
- backpropagation
- linear layers
- activation functions
- multilayer perceptron (regression, classification)

## 10. sequential data
deal with sequential data 

- load, represent and augment sequential data
- gru (sequence classification)
- bert (sequence classification)

## 11. computer vision basics
deal with image data 

- load, represent and augment image data
- convolutional layers
- vgg (image classification)

## 12. advanced training techniques for deep learning
improve the training process of neural networks

- gradient flow theory 
- gradient clipping
- batch normalization
- skip connections (resnet)
- advanced optimizers (rmsprop, adam, adadelta)
- loss weights

## 13. regularization
focus on the signal instead of the noise in your data

- overfitting and underfitting
- adding noise to the model (dropout)
- limiting the range of weights (parameter norm penalties, early stopping, parameter sharing)

## 14. transfer learning
improve neural networks with transfer learning

- fine tuning pretrained models
- multi-task learning
