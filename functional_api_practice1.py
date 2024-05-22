import keras
import numpy as np
from keras import ops
from keras import layers

def func1():
    inputs = keras.Input(shape=(784, ))
    img_inputs = keras.Input(shape=(32, 32, 3))
    print(f"Image shape: {inputs.shape}\nImage type: {inputs.dtype}")
    dense = layers.Dense(units=64, activation='relu')
    x = dense(inputs)
    x = layers.Dense(units=64, activation='relu')(x)
    outputs = layers.Dense(units=10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    model.summary()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(x_train.shape, x_test.shape)
    #print(x_train[0])
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    

    model.compile(
        optimizer = keras.optimizers.RMSprop(),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )
    history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=2, validation_split=0.2)
    test_scores = model.evaluate(x=x_test, y=y_test, verbose=2)
    print('Scores: ', test_scores)
    print('Test Loss: ', test_scores[0])
    print('Test Accuracy: ', test_scores[1])
    print('Model History:\n', history.history)
#func1()

def func2():
    encoder_input = keras.Input(shape=(28, 28, 1), name='img')
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(encoder_input)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)
    encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
    encoder.summary()

    decoder_input = keras.Input(shape=(16,), name='encoded_img')
    x = layers.Reshape(target_shape=(4, 4, 1))(decoder_input)
    x = layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), activation='relu')(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.UpSampling2D(size=(3, 3))(x)
    x = layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), activation='relu')(x)
    decoder_output = layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='relu')(x)
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
    decoder.summary()


    autoencoder_input = keras.Input(shape=(28, 28, 1), name='img')
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoded_img, name='autoencoder')
    autoencoder.summary()
#func2()

def func3():
    def get_model():
        inputs = keras.Input(shape=(128, ))
        outputs = layers.Dense(units=1)(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)
    
    model1 = get_model()
    model2 = get_model()
    model3 = get_model()

    inputs = keras.Input(shape=(128, ))
    y1 = model1(inputs)
    y2 = model2(inputs)
    y3 = model3(inputs)
    outputs = layers.average(inputs=[y1, y2, y3])
    ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
    ensemble_model.summary()
#func3()

def func4():
    num_tags = 12
    num_words = 10000
    num_departments = 4

    title_input = keras.Input(shape=(None, ), name='title')
    body_input = keras.Input(shape=(None, ), name='body')
    tags_input = keras.Input(shape=(num_tags, ), name='tags')

    title_features = layers.Embedding(input_dim=num_words, output_dim=64)(title_input)
    body_features = layers.Embedding(input_dim=num_words, output_dim=64)(body_input)

    title_features = layers.LSTM(units=128)(title_features)
    body_features = layers.LSTM(units=32)(body_features)

    x = layers.concatenate(inputs=[title_features, body_features, tags_input])
    priority_pred = layers.Dense(units=1, name='priority')(x)
    department_pred = layers.Dense(units=num_departments, name='department')(x)

    model = keras.Model(
        inputs = [title_input, body_input, tags_input],
        outputs = {'priority': priority_pred, 'department': department_pred}
    )
    model.summary()
#func4()

def func5():
    inputs = keras.Input(shape=(32, 32, 3), name='img')
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    block_1_output = layers.MaxPooling2D(pool_size=(3, 3))(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    block_2_output = layers.add(inputs=[x, block_1_output])

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    block_3_output = layers.add(inputs=[x, block_2_output])

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    outputs = layers.Dense(units=10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='toy_resnet')
    model.summary()

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print(x_train.shape, y_train.shape)

    x_train = x_train.astype(dtype='float32') / 255.0
    x_test = x_test.astype(dtype='float32') / 255.0
    y_train = keras.utils.to_categorical(x=y_train, num_classes=10)
    y_test = keras.utils.to_categorical(x=y_test, num_classes=10)

    model.compile(
        optimizer = keras.optimizers.RMSprop(1e-3),
        loss = keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )

    history = model.fit(
        x = x_train[:1000],
        y = y_train[:1000],
        batch_size = 64,
        epochs = 2,
        validation_split = 0.2
    )
    print(history.history)
#func5()

def func6():
    vgg19 = keras.applications.VGG19()
    features_list = [layer.output for layer in vgg19.layers]
    example_input = keras.Input(shape=(32, 32, 3), name='example_input')
    #feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
    feat_extraction_model = keras.Model(inputs=example_input, outputs=features_list)
    img = np.random.random(size=(1, 224, 224, 3)).astype(dtype='float32')
    #extracted_features = feat_extraction_model(img)
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(dtype='float32') / 255.0
    x_test = x_test.astype(dtype='float32') / 255.0
    y_train = keras.utils.to_categorical(x=y_train, num_classes=10)
    y_test = keras.utils.to_categorical(x=y_test, num_classes=10)

    feat_extraction_model.compile(
        optimizer = keras.optimizers.RMSprop(1e-3),
        loss = keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )

    history = feat_extraction_model.fit(
        x = x_train[:1000],
        y = y_train[:1000],
        batch_size = 64,
        epochs = 2,
        validation_split = 0.2
    )
    print(history.history)
func6()