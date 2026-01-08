try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from tensorflow.keras import layers, models, Input
except ImportError:
    layers = None
    models = None
    Input = None

    class ModelsModel:
        pass

    models = type("models", (object,), {"Model": ModelsModel})


class OnsetNetV2(models.Model):
    def __init__(self, audio_shape, n_other_features, config):
        super(OnsetNetV2, self).__init__()
        self.config = config

        self.cnn_layers = []
        cnn_filter_shapes = config.get("cnn_filter_shapes", [])
        cnn_pool = config.get("cnn_pool", [])

        for i, (shape, pool) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
            self.cnn_layers.append(
                layers.Conv2D(
                    filters=shape[2],
                    kernel_size=(shape[0], shape[1]),
                    activation="relu",
                    padding="same",
                    name=f"cnn_{i}",
                )
            )
            self.cnn_layers.append(
                layers.MaxPool2D(
                    pool_size=pool, strides=pool, padding="same", name=f"pool_{i}"
                )
            )

        self.flatten = layers.Flatten()

        self.do_rnn = config.get("rnn_size", 0) > 0
        if self.do_rnn:
            self.rnn_proj = layers.Dense(config["rnn_size"])
            rnn_type = config.get("rnn_cell_type", "lstm")
            if rnn_type == "lstm":
                self.rnn = layers.LSTM(config["rnn_size"], return_sequences=True)
            elif rnn_type == "gru":
                self.rnn = layers.GRU(config["rnn_size"], return_sequences=True)
            else:
                self.rnn = layers.SimpleRNN(config["rnn_size"], return_sequences=True)

        self.dnn_layers = []
        dnn_sizes = config.get("dnn_sizes", [])
        for i, size in enumerate(dnn_sizes):
            self.dnn_layers.append(
                layers.Dense(
                    size, activation=config.get("dnn_nonlin", "relu"), name=f"dnn_{i}"
                )
            )
            if config.get("dnn_keep_prob", 1.0) < 1.0:
                self.dnn_layers.append(layers.Dropout(1.0 - config["dnn_keep_prob"]))

        self.logits = layers.Dense(1, name="logits")

    def call(self, inputs, training=False):
        audio_input, other_input = inputs

        x = audio_input
        for layer in self.cnn_layers:
            x = layer(x)

        x = self.flatten(x)
        x = layers.concatenate([x, other_input])

        if self.do_rnn:
            pass  # Placeholder

        for layer in self.dnn_layers:
            x = layer(x, training=training)

        return self.logits(x)


class SymNetV2(models.Model):
    def __init__(self, audio_shape, n_other_features, vocab_size, config):
        super(SymNetV2, self).__init__()
        self.config = config

        embed_size = config.get("sym_embedding_size", 32)
        if embed_size > 0:
            self.embedding = layers.Embedding(vocab_size, embed_size)
        else:
            self.embedding = None

        self.cnn_layers = []
        cnn_filter_shapes = config.get("cnn_filter_shapes", [])
        cnn_pool = config.get("cnn_pool", [])

        for i, (shape, pool) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
            conv = layers.Conv2D(
                filters=shape[2],
                kernel_size=(shape[0], shape[1]),
                activation="relu",
                padding="same",
            )
            maxpool = layers.MaxPool2D(pool_size=pool, strides=pool, padding="same")
            self.cnn_layers.append((conv, maxpool))

        self.cnn_dim_redux = None
        if config.get("cnn_dim_reduction_size", -1) > 0:
            self.cnn_dim_redux = layers.Dense(
                config["cnn_dim_reduction_size"], activation="relu"
            )

        self.rnn_size = config.get("rnn_size", 128)
        self.rnn_layers = []
        n_layers = config.get("rnn_nlayers", 1)
        for i in range(n_layers):
            self.rnn_layers.append(
                layers.LSTM(self.rnn_size, return_sequences=True, return_state=False)
            )

        self.dnn_layers = []
        dnn_sizes = config.get("dnn_sizes", [])
        for i, size in enumerate(dnn_sizes):
            self.dnn_layers.append(layers.Dense(size, activation="relu"))
            if config.get("dnn_keep_prob", 1.0) < 1.0:
                self.dnn_layers.append(layers.Dropout(1.0 - config["dnn_keep_prob"]))

        self.output_layer = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        sym_seq, audio_seq, other_seq = inputs

        if self.embedding:
            x_sym = self.embedding(sym_seq)
        else:
            x_sym = sym_seq

        x_audio = audio_seq
        for conv, pool in self.cnn_layers:
            x_audio = layers.TimeDistributed(conv)(x_audio)
            x_audio = layers.TimeDistributed(pool)(x_audio)

        x_audio = layers.TimeDistributed(layers.Flatten())(x_audio)

        if self.cnn_dim_redux:
            x_audio = layers.TimeDistributed(self.cnn_dim_redux)(x_audio)

        x = layers.concatenate([x_sym, x_audio, other_seq])

        for rnn in self.rnn_layers:
            x = rnn(x, training=training)

        for dnn in self.dnn_layers:
            x = dnn(x, training=training)

        return self.output_layer(x)


def create_onset_model(input_shape, n_other, config):
    audio_in = Input(shape=input_shape, name="audio_in")
    other_in = Input(shape=(n_other,), name="other_in")

    x = audio_in
    cnn_filter_shapes = config.get("cnn_filter_shapes", [])
    cnn_pool = config.get("cnn_pool", [])

    for i, (shape, pool) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
        x = layers.Conv2D(
            filters=shape[2],
            kernel_size=(shape[0], shape[1]),
            activation="relu",
            padding="same",
        )(x)
        x = layers.MaxPool2D(pool_size=pool, strides=pool, padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.concatenate([x, other_in])

    dnn_sizes = config.get("dnn_sizes", [])
    for size in dnn_sizes:
        x = layers.Dense(size, activation=config.get("dnn_nonlin", "relu"))(x)
        if config.get("dnn_keep_prob", 1.0) < 1.0:
            x = layers.Dropout(1.0 - config["dnn_keep_prob"])(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=[audio_in, other_in], outputs=outputs, name="OnsetNet")
    return model


def create_sym_model(seq_len, audio_shape, n_other, vocab_size, config):
    sym_in = Input(shape=(seq_len,), dtype="float32", name="sym_in")
    audio_in = Input(shape=(seq_len,) + audio_shape, name="audio_in")
    other_in = Input(shape=(seq_len, n_other), name="other_in")

    embed_size = config.get("sym_embedding_size", 32)
    if embed_size > 0:
        x_sym = layers.Embedding(vocab_size, embed_size)(sym_in)
    else:
        x_sym = layers.Dense(64, activation="relu")(sym_in)

    x_audio = audio_in
    cnn_filter_shapes = config.get("cnn_filter_shapes", [])
    cnn_pool = config.get("cnn_pool", [])

    for i, (shape, pool) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
        conv = layers.Conv2D(
            filters=shape[2],
            kernel_size=(shape[0], shape[1]),
            activation="relu",
            padding="same",
        )
        pool_layer = layers.MaxPool2D(pool_size=pool, strides=pool, padding="same")

        x_audio = layers.TimeDistributed(conv)(x_audio)
        x_audio = layers.TimeDistributed(pool_layer)(x_audio)

    x_audio = layers.TimeDistributed(layers.Flatten())(x_audio)

    if config.get("cnn_dim_reduction_size", -1) > 0:
        x_audio = layers.TimeDistributed(
            layers.Dense(config["cnn_dim_reduction_size"], activation="relu")
        )(x_audio)

    x = layers.concatenate([x_sym, x_audio, other_in])

    rnn_size = config.get("rnn_size", 128)
    n_layers = config.get("rnn_nlayers", 1)

    for i in range(n_layers):
        x = layers.LSTM(rnn_size, return_sequences=True)(x)

    dnn_sizes = config.get("dnn_sizes", [])
    for size in dnn_sizes:
        x = layers.Dense(size, activation="relu")(x)
        if config.get("dnn_keep_prob", 1.0) < 1.0:
            x = layers.Dropout(1.0 - config["dnn_keep_prob"])(x)

    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = models.Model(
        inputs=[sym_in, audio_in, other_in], outputs=outputs, name="SymNet"
    )
    return model
