Colab svg image link: ![Open In Colab](https://colab.research.google.com/github/mariik4/speed_dating_zneus/blob/feat%2Fafter_analyse_stage/zneus_project.ipynb)

# Analyse

This project is dedicated to the analysis of the Speed Dating Dataset using machine learning techniques. The primary goal is to predict whether two individuals will match based on their attributes and preferences.
We check what data we have, what distribution is there, and what features are important for our prediction task.
Conclusion: We decide to use only some of the features for our model, that are correlated with match target feature.

# First try to model train

We try to train model using all the features that we have after analysing stage.
And we didnt get very good results. All results are provided in baseline report.
We now have baseline to improve our model.
From the results we can see that our model is overfitting a lot, our val_error is growing and our train_error is decreasing, aslo our test_error is increasing.
And to improve it we need to reduce number of features and increase number of samples that we can use for training and also try to tune hyperparameters of our model and try different configurations of model architecture.
Also we want to see one more statistic for our model with validation set.
And we found an issue that we fit_transform not only training set but also validation and test sets. So we fix it in the next try.
Also from the binary classification we can say we want sigmoid activation in the output layer and binary crossentropy loss function. And from our baseline we can see we want less features in first hidden layer.

Our baseline shows that for that classifications we need better hyperparamters, because a lot of models are heavily overfitting. Also we can see that models are very unpredictable on value accuracy. Value accuracy is high, but final test shows accuracy around 0.83. That potentially mean of too high learning rate and very strong overfitting. We should make model smaller.

# Second try to model train

We selected only some of the features that are correlated with match target feature and not with each other.
We also transform nan values using IterativeImputer from sklearn to increase number of samples that we can use for training.
We try to train model using these features:
Also after getting initial results, we can see that model often peaks on 84 accuracy, and overfits. From that and the fact that our data is heavily skewed on match 0, other then 1 (around 80 to 20), we can say that we need to alter our loss function. Also we decided to use sweep to increase number of hyperparameters tested out.
In our sweep model we define hypeparameter to be tested out: different hidden dims (from 4 to 128), batch sizes (From 32 to 512), learning rate (from 0.00001 to 0.01), dropout (from 0.0 to 0.5), activation functions (Gelu, Sigmoid, ReLu, Tanh) and amount of hidden layers (from 2 to 4).
For sweep we will use random method and maximise val_accuracy.

# Third try to model train

We changed loss function by multiplying the target 1 on the proportions of match = 0 to match = 1. In that way model will not predict only zeroes, but will heavily try to predict ones but with less accurate results.
Results of sweep and manual testing showed us that model with 3 and 4 layers are overfitting heavily, and sweep tries to model them with dropout in range of 0.3 to 0.5.
This shows that this models are not efficient. In the next sweeps we will use less 3 layered models and no 4 layers models.
ALso we diagnosed that most of the hidden layers should be RELU or GELU and last layers can be either Sigmoid, or Tanh, or Also RelU. Sigmoid and Tanh functions are underperforming when used multiple times. Also it is possible and efficient to use only ReLU. Models with 2 layers with range of neurons on first layer around 32 - 16 shows good results with less dropout then 0.22. After sweep we fine tune the best models manually changing some of the parameters and watching how test_loss and val_loss performs. If Val_loss raises after some training, we increase drop out, or put less neurons, if it is possible that model underperforms and can still more, we give a little more epochs, and try out bigger learning rate.
After changing the loss function, model lost its accuracy from 0.83 to 0.77-0.79. As we figured out, previous models tended to produce only zeroes, and very rarely or never at all gived answer 1. Thats why model was overall more accurate but useless. New model traing on producing accurate 77% of answers right both on 0 and 1.
