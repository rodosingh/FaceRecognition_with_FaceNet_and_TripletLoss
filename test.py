import pickle
from keras.models import load_model
from helper_func import image_resize_cum_preprocess, compute_dist, split_text
import numpy as np

"""###### From here test.py"""
# Embedder create embedding of given image.
def embedder(img_path, loaded_model):
    """
    Inputs

        image_path: absolute or relative path to image
        loaded_model: the model trained - FaceNet
    
    Outputs

        embedding vector
    """
    prep_image = image_resize_cum_preprocess(img_path, size = (160, 160))
    return loaded_model.predict([prep_image, prep_image, prep_image])[:128]

# Recognise face by comparing the embedding vector against anchor embedding vectors - using L2 norm.
def face_recogniser(model, image_path, threshold, embedding_dict, name):
    """
    Inputs
        image_path: absolute or relative path to image
        threshold: to determine whether the given image matches with existing embeddings
    Outputs
        If embeddings is close to existing ones it return the name of that person or else return unknown.
    """
    
    embeddings = embedder(image_path, model)
    distance_lst = [compute_dist(embeddings, embedding_dict[person]) for person in embedding_dict.keys()]
    print(f"Distance between embedding vectors of anchor and {name}:  {distance_lst}")
    if min(distance_lst) <= threshold:
        print(f"The image provided is of {list(embedding_dict.keys())[np.argmin(distance_lst)]}.")
        return list(embedding_dict.keys())[np.argmin(distance_lst)]
    else:
        print("This image if of unknown person.")
        return "Unknown"

# Now predict all the test images.
def testing_on_test(model, labels_list, embedding_dict, threshold = 0.45):
    k = 0; m = 0
    for image_folder in labels_list['test_name']:
        folder_name = split_text(image_folder[0])
        for image_name in image_folder[1:]:
            img_path = "cropped_images/" + folder_name + '/' + image_name
            pred_name = face_recogniser(model, img_path, threshold = threshold, embedding_dict=embedding_dict, name = folder_name)
            print(f"And the actual owner of this face: {folder_name}")
            print("\n==================================================\n")
            if pred_name == folder_name:
                m += 1
            k += 1
    print(f"Out of total {k} test images, our model have recognized {m} correctly.")
    print("Accuracy of our model: %.3f"%(m*100/k))

# Import train and Test labels.
with open("train_test.txt", "rb") as fp:
    train_test_label = pickle.load(fp)

#Load the embedding dictionary of Anchor Images.
with open("embeddings.txt", "rb") as fpp:
        embedding_dict = pickle.load(fpp)

# load model
model = load_model("FaceNet.h5", compile=True)

# Show on screen whether we are getting correct prediction with threshold = 3.5
testing_on_test(model, train_test_label, embedding_dict, threshold=3.5)


