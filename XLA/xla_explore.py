## Author Manoj Kumar

from ast import arg
from timeit import repeat
from numpy import dtype
import tensorflow as tf
import os
import time
tf.compat.v1.enable_eager_execution()
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

 
# Then define some necessary constants and prepare the MNIST dataset.


# Number of distinct number labels, [0..9]
NUM_CLASSES = 10
samples = 100000
seed=10

# Number of examples in each training batch (step)
TRAIN_BATCH_SIZE = 4
TEST_BATCH=8
epochs = 1
test_iteration = 1

# Number of training steps to run
TRAIN_STEPS = 1000

# Loads CIFAR10 dataset.
#train, test = tf.keras.datasets.cifar10.load_data()


#train_total_samples = train[0].shape[0]
#test_total_samples = test[0].shape[0]
#train_ds = tf.data.Dataset.from_tensor_slices(train).batch(TRAIN_BATCH_SIZE).take(1000).shuffle(train_total_samples, seed=seed).repeat(epochs)

# Casting from raw data to the required datatypes.
def cast(images, labels):
  #images = tf.cast(
  #    tf.reshape(images, [3, IMAGE_SIZE]), tf.float32)
  images = tf.cast(images, tf.float32)
  labels = tf.cast(labels, tf.int64)
  return (images, labels)

model_name = ''
 
# Finally, define the model and the optimizer. The model uses a single dense layer.

def create_vgg16():
    global model_name
    model_name='vgg16'
    basemodel = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3), classes=1000)
    for layer in basemodel.layers:
        layer.trainable = False

    model = tf.keras.models.Sequential([basemodel,     tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)])

    model.compile(optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]) 

    return model

def create_resnet50():
    global model_name
    model_name='resnet50'
    basemodel = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))
    for layer in basemodel.layers:
        layer.trainable = False
        
    model = tf.keras.models.Sequential([basemodel,     tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)])

    model.compile(optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]) 

    return model

def create_resnet101():
    global model_name
    model_name='resnet101'
    basemodel = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3), classes=1000)
    for layer in basemodel.layers:
        layer.trainable = False
        
    model = tf.keras.models.Sequential([basemodel,     tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)])

    model.compile(optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]) 

    return model

def generate_model():
    global model_name
    model_name='simple_model'
    model =  tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(224, (3, 3), padding='same', input_shape=(224,224,3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(224, (3, 3), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]) 
    return model


optimizer = tf.keras.optimizers.Adam()
loss_p = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

jit_compiler=''


def train_dataset(images, labels):
    images, labels = cast(images, labels)

    with tf.GradientTape() as tape:

      predicted_labels = model(images)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=predicted_labels, labels=labels
      ))

      correct_prediction = tf.equal(tf.argmax(predicted_labels, 1), labels)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    layer_variables = model.trainable_variables
    grads = tape.gradient(loss, layer_variables)
    optimizer.apply_gradients(zip(grads, layer_variables))

 
def train_fn():
    print("#### Training Started ####")
    log_once = True
    counter = 0
    train_ds = tf.keras.utils.image_dataset_from_directory(
                os.path.join(args.dir_path, 'training'),
                validation_split=0,
                seed=123,
                image_size=(224, 224),
                batch_size=TRAIN_BATCH_SIZE).repeat(1)

    for images, labels in tqdm(train_ds):
        labels = labels.numpy().reshape(TRAIN_BATCH_SIZE,)
        train_dataset(images, labels)

        """if jit_compiler:
            if log_once:
                print(" --- XLA Enabled")
                log_once = False
            train_cifar_enabled(images, labels)
        else:
            if log_once:
                print(" --- XLA Disabled")
                log_once = False
            train_cifar_disabled(images, labels)"""
        
        counter = counter + 1
        if counter > 10000:
          tf_evaluate(model)
          counter = 0
        
    log_dir='./logdir'
    log_dir = log_dir 

    # ## Save model
    tf.keras.models.save_model(model, './saved_model')
    model.save('./saved_model'+'/model.h5')

@tf.function(jit_compile=True)
def do_inference_enabled(layer, img):
    return layer(img)

@tf.function(jit_compile=False)
def do_inference_disabled(layer, img):
    return layer(img)

def warmup(model):
    print("warmup stage")
    random_gen_img = tf.random.uniform(shape = (1, 224, 224, 3), dtype='float32')
    warmup_itr = 10
    for _ in range(warmup_itr):
        model(random_gen_img)

    return model

def evaluate():
    print("#### Evaluation Started ####")
    model_path = '/home/hno1kor/CodeBase/General/hpc_poc/product/benchmark/XLA/saved_model'
    layer = tf.keras.models.load_model(model_path)

    import numpy as np
    inference_time = []
    confidence = []
    accuracy_col = []
    actual_labels = []

    
    results = pd.DataFrame()

    for itr in range(test_iteration):
        itr_column = 'Itr' + str(itr)
        inference_time.clear()
        confidence.clear()
        accuracy_col.clear()
        actual_labels.clear()

        layer = warmup(layer)

        #train, test = tf.keras.datasets.cifar10.load_data()
        #test_ds = tf.data.Dataset.from_tensor_slices(test).batch(TEST_BATCH).take(5).shuffle(test_total_samples, seed=seed).repeat(1)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(args.dir_path, 'validation'),
            validation_split=0,
            seed=123,
            image_size=(224, 224),
            batch_size=TEST_BATCH).repeat(1)       
        

        print(f"### len = {len(list(test_ds))}")
        for i, (test_img, test_label) in tqdm(enumerate(test_ds), desc=f'{itr_column}'):
            if test_img.shape == (TEST_BATCH, 224, 224, 3):
                test_label = test_label.numpy().reshape(TEST_BATCH,)
                
                img, lbl = cast(test_img, test_label)   

                if args.jit_compile_flag:
                    if args.profile:
                        tf.profiler.experimental.start(model_path + '/profile')
                    start = time.time()
                    predicted_labels = do_inference_enabled(layer, img)
                    end = time.time()
                    if args.profile:
                        tf.profiler.experimental.stop(model_path + '/profile')
                else:
                    if args.profile:
                        tf.profiler.experimental.start(model_path + '/profile')
                    start = time.time()
                    predicted_labels = do_inference_disabled(layer, img)
                    end = time.time()
                    if args.profile:
                        tf.profiler.experimental.stop(model_path + '/profile')

                diff = end-start

                # I got the desired results by changing my code from self.prediction_op = tf.argmax(y_conv, 1) to self.prediction_op = tf.nn.softmax(y_conv)
                # https://stackoverflow.com/questions/38133707/how-can-i-implement-confidence-level-in-a-cnn-with-tensorflow

                inference_time.append(diff/TEST_BATCH)
                confidence.append(tf.nn.softmax(predicted_labels).numpy().max(axis=1))
                correct_prediction = tf.equal(tf.argmax(predicted_labels, 1), lbl)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accuracy_col.append(accuracy.numpy())
                actual_labels.append(lbl.numpy())

            else:
                print("\nIgnore last size", test_img.shape[0])
        
        results[itr_column + "_time"] = inference_time
        results[itr_column+'_acc'] = accuracy_col
        results[itr_column+'_conf'] = confidence
        results[itr_column + '_aclabel'] = actual_labels

    #dataframe_name = 'XLA_' + folderStatusName[1:] + '_' + model_name + '_bch' + str(TEST_BATCH) + '.csv'
    dataframe_name = f'XLA_{folderStatusName[1:]}_{model_name}_bch{str(TEST_BATCH)}.csv'
    
    results.drop(axis=1, inplace=True, index=0)
    results.to_csv(dataframe_name)
    print("Data is saved in ", dataframe_name)
    get_stats(dataframe_name)

def get_stats(csv_file):
    trace_d = pd.read_csv(csv_file)
    print(trace_d.index)
    trace_d.drop(axis=1, inplace=True, index=0)
    time_columns = [col for col in trace_d.columns if 'time' in col]
    mean_time = trace_d[time_columns].mean().mean()
    print(f"Inference took {mean_time}s, {folderStatusName[1:]}")

def tf_evaluate(run_time_eval_model = None):

    test_ds = tf.keras.utils.image_dataset_from_directory(
            "/home/hno1kor/Downloads/Dataset/validation",
            validation_split=0,
            seed=123,
            image_size=(224, 224),
            batch_size=TRAIN_BATCH_SIZE).repeat(1)

    if run_time_eval_model == None:
        model_path = './saved_model' + '/model.h5'
        print(model_path)
        layer = tf.keras.models.load_model(model_path)
    else:
        print("Evaluating passed model")
        layer = run_time_eval_model

    #images, labels = cast(train[0], train[1])

    layer.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])


    #images, labels = cast(test[0], test[1])
    loss, acc = layer.evaluate(test_ds)
    print("tf_evaluate model, Test accuracy: {:5.2f}%".format(100 * acc))



if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit-compile-flag', action='store_true', default=False, help="Set true to enable jit_compiler")
    parser.add_argument('--infe-batch', default=1, help="Set the batch size used in inference", type=int)
    parser.add_argument('--dir-path', default='./Dataset', help="Set dataset path", type=str)
    parser.add_argument('--only-train', action='store_true', default=False, help="Train to save model")
    parser.add_argument('--create-model', default='resnet50', help='set which model to use for inference')
    parser.add_argument('--profile', action='store_true', default=False, help='Enable profiling inference code')
    args = parser.parse_args()

    model_fn = {'resnet50':create_resnet50,
                'resnet101':create_resnet101,
                'vgg16':create_vgg16,
                'ownmodel':generate_model
                }

    jit_compiler = args.jit_compile_flag
    TEST_BATCH = args.infe_batch
    print('Jit Compiler = ', jit_compiler)

    if jit_compiler:
        folderStatusName = '/jit_enabled'
    else:
        folderStatusName = '/jit_disabled'
    
    model = model_fn[args.create_model]()
    print("Using Moldel {}",model_name)

    if args.only_train:
        train_fn()
        tf_evaluate()
    else:
        evaluate()