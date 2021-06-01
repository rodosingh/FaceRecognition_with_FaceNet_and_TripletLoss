# FaceRecognition_with_FaceNet_and_TripletLoss
* Perform Face Recognition with FaceNet on Avengers __Chris Evans__, __Mark Ruffalo__, __Chris Hemsworth__, __Robert Downey Jr__, __Scarlett Johansson__.

*Hi everyone, let's walkthrough this project of Face Recognition.*
- I'm sorry for not able to solve the problem persisting with __Docker push and pull__ (_Unsolved 'tensorflow.python.keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'_), as I'm very new to Docker. Today only I have viewed some lectures to implement this.
- But, the code is all fine. And I will let you know what are the __Dependencies__ required for this task and what steps we to follow to train our model and predict all the images in __test__ set of `20%` volume.

## Overview/ Approach

1. I have used pretrained FaceNet model (available at [this link by Hiroki Taniai](https://github.com/nyoki-mtl/keras-facenet) trained on [MS-Celeb-1M dataset](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) with model weights and architecture available at [here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)). This model has input dimension of `160 * 160`, while the output is an embedding vector of size `128`.
2. Used __Triplet Loss__ as the loss function with `margin = 0.9`.
3. Except the last `Convolution2D` block, I freezed all the blocks while trainng so that the weights can get acclimatized to the new data through some fine changes in weights of top block where complex features are learned (also callled __fine tuning__).
4. Beforehand, we chose one image from each class as __Anchor__ image and created their embeddings in `train.py`. This way we would prevent computing the embeddings of those images while performing face recognition of some test images.
5. While testing for the _test set_, we compute the __euclidean distance__ of embedding vector of each test image from the embedding vectors we have in our __Anchor__ dictionary. If the lowest distance among all those 5 classes is less than some threshold (in our case it is `threshold = 3.5`), then the class having the lowest distance from the given test embedding vector is assigned to that vector (or person's image).

## Install the dependencies

```python3
pip install -r requirements.txt
```

## How to run?

1. Please find the Dockerfile at [here](Dockerfile) where everything along with a [bash script](script.sh), that run [`train.py`](train.py) and [`test.py`](test.py) sequentially, is being added to a docker image .
> Disclaimer: But for some reason, `keras` is not installing properly for which `train.py` throws some error. But we can manually do it as follows.

2. In [`train.py`](train.py) we have code to access GPU with `90%` capacity as we want to prevent `OOM Error`.
3. Open your terminal in the current directory and type `python3 train.py`. Make sure that you have installed the [requirements](requirements.txt).
4. The terminal will prompt you to enter `batch_size` (take it 128) and `no. of epochs` (take it to 20).
5. Now run `python3 test.py` and see the predictions for each test image with _total accuracy_ at end.

## About other files/ function.

* [`helper_func.py`](helper_func.py) contains those functions that we use in both [`train.py`](train.py) and [`test.py`](test.py).
* [labelled_pkl](labelled_pkl) cotnains the preprocessed trainng data with their labels in class-wise fashion.
* [train_test.txt](train_test.txt) contains the name of images that fall in training set and test set in `80:20` ratio. Also we can generate our own set of train and test just by uncommenting the code in [`train.py`](train.py) at line __83__ and saving it using line __86__.
* `triplet_loss()` function with margin `0.9` is responsible for our algorithm `FaceNet` training.
* `get_batch_random()` is again a nice function to generate triplets from the data ([labelled_pkl](labelled_pkl)).

___See the below gif file to have a taste of our implementation.___

> ![Face_recognition_Walkthrough](face_recognition.gif)

