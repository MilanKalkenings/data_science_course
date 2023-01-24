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

## 2. data cleaning and feature engineering
clean data and engineer features for analysis with pandas 

- data frames (creation, shape, head)
- dropping (drop rows, drop columns)
- indices (change column ids, change row ids)
- indexing and slicing (columns and rows)
- masking (return rows that fulfill certain conditions)
- changing values in columns
- sorting (sort_values())
- counting (value_counts())
- null values (isna(), dropna(), fillna())
- uniques
- groupby
- appending new columns (pd.concat())
- creating new columns from existing ones 

tbd 
- merge data frames

## 3. exploratory data analysis with pandas and matplotlib
means of statistics to gain insights into your data and to test your assumptions

- summary statistics (mean, median, std, var)
- normal distribution
- pearson correlation
- hypothesis testing
- boxplot
- histplot
- countplot
- violinplot
- heatmap
- scatterplot
- lineplot

## 4. machine learning basics with sklearn
machine learning to mine patterns in your data and to solve a variety of problems

- train / val / test split
- linear regression
- logistic regression
- decision tree
- kmeans
- hyperparameter optimization

## 5. machine learning ensembles
machine learning ensembles to achieve better results

- error decorrelation
- averaging
- bagging (random forest)
- boosting (xgboost)

## 6. git
share your code with others and benefit from version control

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

- torch tensors
- loss functions
- stochastic gradient descent
- backpropagation
- linear layers
- activation functions
- multilayer perceptron (regression, classification)

## 10. computer vision basics
deal with image data 

- load image data
- display image data 
- convolutional layers
- alexnet (image classification)

## 11. sequential data
deal with sequential data 

- handling sequential data
- lstm (sequence classification)
- bert (sequence classification)

## 12. advanced training techniques for deep learning
improve the training process of neural networks

- gradient flow theory 
- gradient clipping
- skip connections (resnet)
- advanced optimizers (rmsprop, adam, adadelta)
- loss weights

## 13. regularization
focus on the signal instead of the noise in your data

- overfitting and underfitting
- parameter norm penalties
- dropout
- data augmentation (smote, image augmentation, text augmentation)

## 14. meta learning
improve neural networks with meta learning

- fine tuning pretrained models
- multi-task learning
