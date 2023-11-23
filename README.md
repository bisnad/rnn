# Machine Learning for Movement Continuation #

Daniel Bisig - Instituto Stocos, Spain - daniel@stocos.com, Zurich University of the Arts, Switzerland - daniel.bisig@zhdk.ch

### Overview ###

This repository provides the source code to train a machine learning model on recorded dance movements in the form of motion capture data. This model can be used to predict how a given dance movement could continue over the next few minutes. The model is an recurrent neural network that is implemented using the Pytorch development framework. Recurrent neural networks belongs to a class of models that can learn temporal sequences of data.  

Apart from source code, the repository also includes the weights of a model that was pretrained on several different motion capture recordings. The code has been tested on Windows and MacOS. Anaconda environments for these two operating systems are provided as part of this repository. 