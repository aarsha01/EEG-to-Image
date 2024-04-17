import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from lstm_kmean.model import TripleNet
from model import DCGAN
from tqdm import tqdm
import os
import math
import re

# Function to preprocess data
def preprocess_data(X, Y, P, resolution=128):
    # Preprocess X
    X = tf.squeeze(X, axis=-1)
    max_val = tf.reduce_max(X) / 2.0
    X = (X - max_val) / max_val
    X = tf.transpose(X, [1, 0])
    X = tf.cast(X, dtype=tf.float32)
    # Preprocess Y
    Y = tf.argmax(Y)
    # Preprocess image
    I = tf.image.decode_jpeg(tf.io.read_file(P), channels=3)
    I = tf.image.resize(I, (resolution, resolution))
    I = (tf.cast(I, dtype=tf.float32) - 127.5) / 127.5
    return X, Y, I

# Function to generate images
def generate_images(X, Y, test_path):
    n_classes=10
    latent_dim=128
    input_res=128
    X_preprocessed, Y_preprocessed, I_preprocessed = preprocess_data(X, Y, test_path)
    X_preprocessed = np.expand_dims(X_preprocessed, axis=0)
    triplenet = TripleNet(n_classes=n_classes)
    triplenet_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet)
    triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='lstm_kmean/experiments/best_ckpt', max_to_keep=5000)
    triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
    _, latent_Y = triplenet(X_preprocessed, training=False)
    test_eeg_cls = {}
    Y_preprocessed = Y_preprocessed.numpy()
    if Y_preprocessed not in test_eeg_cls:
        test_eeg_cls[Y_preprocessed] = [np.squeeze(triplenet(X_preprocessed, training=False)[1].numpy())]
    else:
        test_eeg_cls[Y].append(np.squeeze(triplenet(X_preprocessed, training=False)[1].numpy()))
    test_image_count = 1
    test_eeg_cls[Y_preprocessed] = np.array(test_eeg_cls[Y_preprocessed])
    N = test_eeg_cls[Y_preprocessed].shape[0]
    per_cls_image = int(math.ceil((test_image_count / N)))
    test_eeg_cls[Y_preprocessed] = np.expand_dims(test_eeg_cls[Y_preprocessed], axis=1)
    test_eeg_cls[Y_preprocessed] = np.tile(test_eeg_cls[Y_preprocessed], [1, per_cls_image, 1])
    test_eeg_cls[Y_preprocessed] = np.reshape(test_eeg_cls[Y_preprocessed], [-1, latent_dim])

    lr = 3e-4
    model = DCGAN()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory='experiments/best_ckpt', max_to_keep=300)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    START = int(ckpt.step.numpy())

    gpus = tf.config.list_physical_devices('GPU')
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:1'], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    n_gpus = mirrored_strategy.num_replicas_in_sync

    test_noise = np.random.uniform(size=(test_eeg_cls[Y_preprocessed].shape[0], 128), low=-1, high=1)
    noise_lst = np.concatenate([test_noise, test_eeg_cls[Y_preprocessed]], axis=-1)
    print(noise_lst)
    save_path = 'experiments/for_gui/{}/{}'.format(210, Y_preprocessed)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for idx, noise in enumerate(tqdm(noise_lst)):
        X = mirrored_strategy.run(model.gen, args=(tf.expand_dims(noise, axis=0),))
        X = cv2.cvtColor(tf.squeeze(X).numpy(), cv2.COLOR_RGB2BGR)
        X = np.uint8(np.clip((X * 0.5 + 0.5) * 255.0, 0, 255))
        cv2.imwrite(save_path + '/{}_{}.jpg'.format(Y_preprocessed, idx), X)

    image_files = os.listdir(save_path)
    
    st.subheader(":blue[Generated Images:]")
    for image_file in image_files:
        image_path = os.path.join(save_path, image_file)
        # Display each image
        st.image(image_path, caption=image_file, use_column_width="never",width=200)


# Main Streamlit app
def main():
    # Set wallpaper background and title color
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpaperbat.com/img/161069-neural-network-wallpaper.gif");
             background-size: cover;
         }}
         .title {{
             color: white;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    st.markdown("<h1 class='title'>Image Generation using Brain waves</h1>", unsafe_allow_html=True)
    
    # Text inputs for X and Y
    x_input = st.text_area(":blue[Enter X]", value="", height=50)
    # y_input = [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]

    # Button to generate images
    if st.button("Generate Images"):
        # Preprocess X
        x_values = eval(x_input)
        X = np.array(x_values)
        # Preprocess Y
        
        Y = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
        # Test image path
        test_path = "data/images/test/rose/n12620196_45702.JPEG"
        # Generate images and display them
        generate_images(X, Y, test_path)
        st.success(":Images generated successfully!")

if __name__ == "__main__":
    main()