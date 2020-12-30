3#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: future_fstrings -*-
#
# tf_sentiment.py
# Mal Minhas, <mal@malm.co.uk>
#
# Description
# -----------
# The starting point for this script is the following YouTube tutorial:
# How to do Sentiment Analysis with TensorFlow 2
# https://www.youtube.com/watch?v=lKoPIQdRBME&ck_subscriber_id=979636326
#
# Notes
# -----
# A useful detailed explanation to explain what's going on is available here:
# https://www.tensorflow.org/tutorials/text/text_classification_rnn
# Some rough notes:
# What are we doing here?
# We are training an RNN on the IMDB large movie review dataset for sentiment analysis.
# The IMDB dataset is a binary classifications dataset.
# It contains 25k of highly polar reviews for training, And 25k of reviews for testing.
# The 'imdb_reviews/subwords8k' dataset is already tokenized
# Therefore you can use it to create a word embedding layer directly as the input to model. 
# The resulting model is able to distinguish between positive and negative reviews.
# In more specific detail, the steps are:
#
# 1. Load the model
# We load the model using tfds.load() convenience method that returns a tuple of:
# a. 'dict' with keys [train,test,unsupervised] of underlying IMDB datasets
# b. 'tensorflow_datasets.core.dataset_info.DatasetInfo' holding dataset metadata
# Each dataset is a collection of tf.Tensor tuples containing text plus binary label
# We can inspect the underlying raw text using encoder.decode
#
# 2. Build the model
# Basic model has embedding input layer + Bidirecctional LSTM + Dense layer + Logit output
#
# 3. Save the outputs
# We can dump out a graph of the accuracy and loss during building.  Also save the model
#
# 4. Apply the model
# We can apply the model to some sample input reviews. This input will need to be encoded.
#

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

PROGRAM         = __file__
VERSION         = '0.2'
AUTHOR          = 'mal@malm.co.uk'
DATE            = '23.12.20'

def configure_dataset(dataset):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)

def loadDataset(dataset):
    print(f'::loadDataset("{dataset}")')
    #dataset = configure_dataset(dataset)
    dataset, info = tfds.load(dataset, with_info=True, as_supervised=True)
    # encoder = reduced dimensional representation of a set of words => word embedding
    # Associated with an n dimension vector.
    print(f'dataset type,keys:\n{type(dataset)}')
    print(dataset.keys())
    print(f'info features:\n{info.features}')
    encoder = info.features['text'].encoder
    print(f'encoder = {encoder}')
    return info, encoder, dataset

def constructModelSimple(encoder):
    print(f'::constructModelSimple()')
    print(f'Sequential Keras 4 layer model - INPUT/embedding, LSTM64, Dense relu, OUTPUT/sigmoid/logit')
    return tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

def constructModelComplex(encoder):
    print(f'::constructModelComplex()')
    print(f'Sequential Keras 6 layer model - INPUT/embedding, LSTM64, LSTM32, Dense relu, dropout, OUTPUT/sigmoid/logit')
    return tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')])

def dumpSamples(dataset, encoder, n=4):
    print(f'::dumpSamples(n={n})')
    train_dataset = dataset['train']
    print(f'element_spec={type(train_dataset.element_spec)}')
    ds = dataset['train'].take(n)
    for text,label in ds:
        # type will be tf.Tensor, tf.Tensor
        print(type(text),type(label))
        v_text = encoder.decode(text)
        v_label = label.numpy()
        getSentiment = lambda v: v and 'POSITIVE' or 'NEGATIVE'
        print(v_text, getSentiment(v_label))
        print('---------------------')

def fitModel(model, dataset, encoder, nepochs, buffer_size, batch_size, nvalidation):
    print(f'::fitModel()')
    padded_shapes = ([None], [])
    dumpSamples(dataset, encoder)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_dataset = train_dataset.shuffle(buffer_size).padded_batch(batch_size, padded_shapes=padded_shapes).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=padded_shapes).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    # History is only useful for plotting purposes during building showing accuracy going up and loss going down.
    history = model.fit(train_dataset, epochs=nepochs, validation_data=test_dataset, validation_steps=nvalidation)
    test_loss, test_acc = model.evaluate(test_dataset)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    return history

def loadModel(filepath):
    print(f'::loadModel()')
    model = tf.keras.models.load_model(filepath)
    return model

def saveModel(model, filepath):
    print(f'::saveModel("{filepath}")')
    tf.keras.models.save_model(
        model, filepath, overwrite=True, include_optimizer=True, save_format=None,
        signatures=None, options=None, save_traces=True
    )

def plot_graphs(history, metric):
    print(metric)
    print(dir(history.history))
    print(history.history[metric])

    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def plot_accuracy_loss(history, filepath):
    print(f'::plot_accuracy_loss("{filepath})"')
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None,1)
    plt.subplot(1,2,2)
    plot_graphs(history, 'loss')
    plt.ylim(0,None)
    plt.savefig(filepath, format='png')

def pad_to_size(vec, size):
    # pad with zeros
    zeros = [0]*(size-len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad, model):
    # we can't just do model.predict beccause of needing to deal with padding.
    encoded_sample_pred_text = encoder.encode(sentence)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    return predictions

def procArguments(args):
    model = args.get('<model>')
    nepochs = args.get('<nepochs>') or 5
    return model, nepochs

def runTests(model):
    print(f'---------- test model ------------')
    samples = [('This move was awesome.  The acting was incredible.  Highly recommend!'),
               ('This move was so so.  The acting was mediocre.  Kind of recommend')]
    for sample_text in samples:
        predictions = sample_predict(sample_text, pad=True, model=model) * 100
        print(f'"{sample_text}"')
        print('\tProbability this is a positive review %.2f' % predictions)

if __name__ == '__main__':
    import time
    import docopt
    usage="""
    {}
    --------------------------
    Usage:
    {} <model> [<nepochs>] [-v] [-f]
    {} -h | --help
    {} -v | --verbose
    {} -V | --version
    {} -f | --force

    Options:
    -h --help               Show this screen.
    -v --verbose            Verbose mode.
    -V --version            Show version.
 
    Examples:
    1. Run sentiment analysis wwith simple LSTM model:
    {} simple 
    2. Run sentiment analysis wwith multilayer LSTM model:
    {} simple 
    3. Force rebuild of simple LSTM model:
    {} simple -f

    """.format(*tuple([PROGRAM] * 9))

    arguments = docopt.docopt(usage)
    #print(arguments)
    verbose = False
    slack = False
    force = False
    t0 = time.time()
    if arguments.get('--verbose') or arguments.get('-v'):
        verbose = True
    if arguments.get('--force') or arguments.get('-f'):
        force = True
    if arguments.get('--version') or arguments.get('-V'):
        print(f'{PROGRAM} version {VERSION} {AUTHOR} {DATE}')
    elif arguments.get('--help') or arguments.get('-h'):
        print(usage)
    else:
        model, nepochs = procArguments(arguments)
        if model == 'simple':
            modelType = 'singleLSTM' + '.mod'
        else:
            modelType = 'dualLSTMwithDropout' + '.mod'
        if force:
            print(f'WARNING: overwriting existing model "{modelType}"')
        buffer_size = 10000
        batch_size = 64
        nvalidation = 30

        print(f'---------- build model ------------')
        sourcedataset = 'imdb_reviews/subwords8k'  # already tokenized
        #sourcedataset = 'imdb_reviews'  # gets you the raw text reviews
        t0 = time.time()
        print(f'1. Load dataset "{sourcedataset}"')
        info, encoder, dataset = loadDataset(sourcedataset)
        if os.path.exists(modelType) or not force:
            print(f'2. Load model from "{modelType}"')
            model = loadModel(modelType)
            print(model.summary())
            t1 = time.time()
            runTests(model)
            print(f'=== Fininshed running existing {modelType} model in {round(t1-t0, 2)} seconds')
        else:
            print(f'2. Construct model "{modelType}"')
            if modelType == 'singleLSTM':
                model = constructModelSimple(encoder)
            else:
                model = constructModelComplex(encoder)
            print(f'3. Fit model "{modelType}"')
            history = fitModel(model, dataset, encoder, nepochs, buffer_size, batch_size, nvalidation)
            plot_accuracy_loss(history, f'{modelType}_accuracyLoss.png')
            print(f'4. Save model "{modelType}"')
            saveModel(model, modelFile)
            runTests(model)
            t1 = time.time()
            t = round((t1-t0)/60)
            print(f'=== Fininshed building {modelType} model with {nepochs} epochs and {nvalidation} validations in {t1-t0} minutes')
    