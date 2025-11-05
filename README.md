Colab svg image link: ![Open In Colab](https://colab.research.google.com/github/mariik4/speed_dating_zneus/blob/feat%2Fafter_analyse_stage/zneus_project.ipynb)

# Analyse
This project is dedicated to the analysis of the Speed Dating Dataset using machine learning techniques. The primary goal is to predict whether two individuals will match based on their attributes and preferences.
We check what data we have, what distribution is there, and what features are important for our prediction task.
We decide to use only some of the features for our model, that are  correlated with match target feature.
Conclusion: we decide to use only this columns for our model: ...

# First try to model train
We try to train model using all the features that we have after analysing stage.
And we didnt get very good results. All results are provided in baseline report.
We now have baseline to improve our model.
From the results we can see that our model is overfitting a lot, our val_error is growing and our train_error is decreasing, aslo our test_error is increasing.TODO: (check if im correct here!!!!)
And to improve it we need to reduce number of features and increase number of samples that we can use for training and also try to tune hyperparameters of our model and try different configurations of model architecture.
Also we want to see one more statistic for our model with validation set.
And we found an issue that we fit_transform not only training set but also validation and test sets. So we fix it in the next try.
Also from the binary classification we can say we want sigmoid activation in the output layer and binary crossentropy loss function. And from our baseline we can see we want less features in first hidden layer.
[[Baseline Report]]


Our baseline shows
TODO: 
# Second try to model train
We selected only some of the features that are correlated with match target feature and not with each other.
We also transform nan values using IterativeImputer from sklearn to increase number of samples that we can use for training.
We try to train model using these features: ...