import keras

def feedforward_1(input_shape = None, num_labels = None):
    if input_shape is None:
        raise Exception('input_shape must be provided as a tuple, e.g., (784,)')
    if num_labels is None:
        raise Exception('num_labels must be provided as an integer')
    inputs = keras.Input(shape = input_shape)
    x = inputs
    x = keras.layers.GaussianNoise(stddev = 0.2)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(num_labels)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model

def feedforward_2(input_shape = None, num_labels = None):
    if input_shape is None:
        raise Exception('input_shape must be provided as a tuple, e.g., (784,)')
    if num_labels is None:
        raise Exception('num_labels must be provided as an integer')
    inputs = keras.Input(shape = input_shape)
    x = inputs
    layer_size = 512
    for l in range(6):
      x = keras.layers.Dense(layer_size, activation = 'linear',
                             kernel_constraint = keras.constraints.max_norm(1.0))(x)
      x = keras.layers.BatchNormalization()(x) # here was an error
      x = keras.layers.Activation('relu')(x)
    outputs = keras.layers.Dense(num_labels, activation = "softmax", # here was an error
                                 kernel_constraint = keras.constraints.max_norm(1.0))(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = 1.0e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.summary()
    return model

def feedforward_3(input_shape = None, num_labels = None):
    if input_shape is None:
        raise Exception('input_shape must be provided as a tuple, e.g., (784,)')
    if num_labels is None:
        raise Exception('num_labels must be provided as an integer')
    inputs = keras.Input(shape = input_shape)
    x = inputs
    layer_size = 512
    for l in range(8):
      x = keras.layers.Dense(layer_size, activation = 'linear',
                             kernel_constraint = keras.constraints.max_norm(1.0))(x)
      x = keras.layers.Activation('relu')(x)
      x = keras.layers.Dropout(rate = 0.5)(x)

    outputs = keras.layers.Dense(num_labels, activation = "softmax", # here was an error
                                 kernel_constraint = keras.constraints.max_norm(1.0))(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = 1.0e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.summary()
    return model

def cnn_1(input_shape = None, num_labels = None):
    if input_shape is None:
        raise Exception('input_shape must be provided as a tuple, e.g., (28, 28)')
    if num_labels is None:
        raise Exception('num_labels must be provided as an integer')
    inputs = keras.Input(shape = input_shape)
    x = inputs
    #
    filter_sizes = [64, 128]
    #
    for fs in filter_sizes:
      x = keras.layers.Conv2D(fs, kernel_size = (3, 3),
                              activation = None,
                              kernel_constraint = keras.constraints.max_norm(1.0))(x)
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.Activation('relu')(x)
      x = keras.layers.MaxPooling2D(pool_size = (2, 2),
                                    strides = (2, 2),
                                    padding = 'valid')(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(num_labels,
                                 kernel_constraint = keras.constraints.max_norm(3.0))(x) # ACTIVATION NOT BECAUSE FROM LOGITS = True
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model

def cnn_2(input_shape = None, num_labels = None):
    if input_shape is None:
      raise Exception('input_shape must be provided as a tuple, e.g., (28, 28)')
    if num_labels is None:
      raise Exception('num_labels must be provided as an integer')
    inputs = keras.Input(shape = input_shape)
    x = inputs
    #
    filter_sizes = [64, 128]
    #
    for fs in filter_sizes:
      temp_layers = list()
      for i in range(4):
        x = keras.layers.Conv2D(fs, kernel_size = (3, 3),
                                activation = None,
                                padding = 'same',
                                kernel_constraint = keras.constraints.max_norm(1.0))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        temp_layers.append(x)
      x = keras.layers.Concatenate()(temp_layers)
      x = keras.layers.MaxPooling2D(pool_size = (2, 2),
                                    strides = (2, 2),
                                    padding = 'valid')(x)
    #
    x = keras.layers.Flatten()(x)
    #
    outputs = keras.layers.Dense(num_labels,
                                 kernel_constraint = keras.constraints.max_norm(1.0),
                                 activation = 'softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.summary()
    return model

def cnn_3(input_shape = None, num_labels = None):
  if input_shape is None:
    raise Exception('input_shape must be provided as a tuple, e.g., (5, 21, 14)')
  if num_labels is None:
    raise Exception('num_labels must be provided as an integer')
  inputs = keras.Input(shape = input_shape)
  x = inputs
  x = keras.layers.Conv2D(64, kernel_size = (2, 21))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Flatten()(x)
  #
  x = keras.layers.Dense(1024)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  #
  x = keras.layers.Dense(1024)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  #
  outputs = keras.layers.Dense(num_labels,
                               kernel_constraint = keras.constraints.max_norm(1.0),
                               activation = 'softmax')(x)
  model = keras.Model(inputs, outputs)
  model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
  )
  model.summary()
  return model

def rnn_1(input_shape = None, num_labels = None):
    if input_shape is None:
      raise Exception('input_shape must be provided as a tuple, e.g., (10, 28)')
    if num_labels is None:
      raise Exception('num_labels must be provided as an integer')
    print(input_shape)
    inputs = keras.Input(shape = input_shape)
    x = inputs
    #
    x = keras.layers.LSTM(512,
                          kernel_constraint = keras.constraints.max_norm(3.0),
                          recurrent_constraint = keras.constraints.max_norm(3.0),
                          dropout = 0.5,
                          recurrent_dropout = 0.5,
                          activation = 'relu',
                          return_sequences = False)(x)
    #
    x = keras.layers.Dense(512, kernel_constraint = keras.constraints.max_norm(3.0))(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(rate = 0.5)(x)
    #
    outputs = keras.layers.Dense(num_labels,
                                 kernel_constraint = keras.constraints.max_norm(3.0),
                                 activation = 'softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = 1.0e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.summary()
    return model

def rnn_2(input_shape = None, num_labels = None):
    if input_shape is None:
      raise Exception('input_shape must be provided as a tuple, e.g., (10, 28)')
    if num_labels is None:
      raise Exception('num_labels must be provided as an integer')
    inputs = keras.Input(shape = input_shape)
    x = inputs
    #
    layer_size = 128
    #
    x = keras.layers.Bidirectional(keras.layers.LSTM(layer_size,
                          kernel_constraint = keras.constraints.max_norm(3.0),
                          recurrent_constraint = keras.constraints.max_norm(3.0),
                          dropout = 0.5,
                          recurrent_dropout = 0.5,
                          activation = 'relu',
                          return_sequences = True))(x)
    #
    x = keras.layers.Bidirectional(keras.layers.LSTM(layer_size,
                          kernel_constraint = keras.constraints.max_norm(3.0),
                          recurrent_constraint = keras.constraints.max_norm(3.0),
                          dropout = 0.5,
                          recurrent_dropout = 0.5,
                          activation = 'relu',
                          return_sequences = False))(x)
    #
    x = keras.layers.Dense(layer_size, kernel_constraint = keras.constraints.max_norm(3.0))(x)
    x = keras.layers.Dropout(rate = 0.5)(x)
    x = keras.layers.Activation('relu')(x)
    #
    outputs = keras.layers.Dense(num_labels,
                                 kernel_constraint = keras.constraints.max_norm(1.0),
                                 activation = 'softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = 1.0e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.summary()
    return model