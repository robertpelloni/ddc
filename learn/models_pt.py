import torch
import torch.nn as nn
import torch.nn.functional as F


class OnsetNet(nn.Module):
    def __init__(self, audio_shape, n_other, config):
        """
        audio_shape: (channels, time_steps, freq_bins) - Note: Channel first for PyTorch
        """
        super(OnsetNet, self).__init__()
        self.config = config

        self.cnn_layers = nn.ModuleList()
        cnn_filter_shapes = config.get(
            "cnn_filter_shapes", [(3, 3, 16), (3, 3, 32), (3, 3, 64)]
        )
        cnn_pool = config.get("cnn_pool", [(1, 2), (1, 2), (1, 2)])

        in_channels = audio_shape[0]  # Expecting 3

        for i, (shape, pool) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
            # shape is (kernel_h, kernel_w, out_channels)
            out_channels = shape[2]
            kernel_size = (shape[0], shape[1])

            self.cnn_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
            )
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(
                nn.MaxPool2d(kernel_size=pool, stride=pool, padding=0)
            )  # Padding handled via functional or manual if needed, but 'same' in Conv usually suffices for shape. PyTorch MaxPool padding is different.
            # Note: TF 'same' padding in MaxPool is tricky. Let's assume input sizes are sufficient.
            # Actually, standard DDC uses fixed input sizes.

            in_channels = out_channels

        # Calculate Flatten size
        # We'll do a dummy pass in __init__ to determine linear layer size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *audio_shape)
            x = dummy_input
            for layer in self.cnn_layers:
                x = layer(x)
            self.flatten_size = x.numel()

        self.dnn_layers = nn.ModuleList()
        dnn_sizes = config.get("dnn_sizes", [128, 64])

        input_dim = self.flatten_size + n_other

        for size in dnn_sizes:
            self.dnn_layers.append(nn.Linear(input_dim, size))
            self.dnn_layers.append(nn.ReLU())
            dropout_prob = 1.0 - config.get("dnn_keep_prob", 1.0)
            if dropout_prob > 0:
                self.dnn_layers.append(nn.Dropout(dropout_prob))
            input_dim = size

        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, audio_input, other_input):
        x = audio_input
        for layer in self.cnn_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, other_input], dim=1)

        for layer in self.dnn_layers:
            x = layer(x)

        return torch.sigmoid(self.output_layer(x))


class SymNet(nn.Module):
    def __init__(self, audio_shape, n_other, vocab_size, config):
        """
        audio_shape: (channels, time_steps, freq_bins)
        """
        super(SymNet, self).__init__()
        self.config = config

        # Embedding
        embed_size = config.get("sym_embedding_size", 32)
        if embed_size > 0:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            sym_feat_size = embed_size
        else:
            self.embedding = None
            self.sym_dense = nn.Linear(
                1, 64
            )  # Assuming scalar input if no embedding? Actually input is indices.
            # In TF code: layers.Dense(64)(sym_in) where sym_in is (batch, seq).
            # This implies one-hot or just projection. But typically embedding is used.
            # We will assume embedding is always used for discrete tokens.
            sym_feat_size = 64

        # CNN for Audio (Time Distributed manually)
        self.cnn_layers = nn.ModuleList()
        cnn_filter_shapes = config.get(
            "cnn_filter_shapes", [(3, 3, 16), (3, 3, 32), (3, 3, 64)]
        )
        cnn_pool = config.get("cnn_pool", [(1, 2), (1, 2), (1, 2)])

        in_channels = audio_shape[0]

        for i, (shape, pool) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
            out_channels = shape[2]
            kernel_size = (shape[0], shape[1])

            self.cnn_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
            )
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
            in_channels = out_channels

        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *audio_shape)
            x = dummy_input
            for layer in self.cnn_layers:
                x = layer(x)
            self.cnn_flatten_size = x.numel()

        self.cnn_dim_reduction = None
        cnn_out_size = self.cnn_flatten_size
        if config.get("cnn_dim_reduction_size", -1) > 0:
            self.cnn_dim_reduction = nn.Linear(
                self.cnn_flatten_size, config["cnn_dim_reduction_size"]
            )
            cnn_out_size = config["cnn_dim_reduction_size"]

        # RNN
        rnn_input_size = sym_feat_size + cnn_out_size + n_other
        rnn_size = config.get("rnn_size", 128)
        num_layers = config.get("rnn_nlayers", 1)

        self.rnn = nn.LSTM(
            rnn_input_size, rnn_size, num_layers=num_layers, batch_first=True
        )

        # DNN
        self.dnn_layers = nn.ModuleList()
        dnn_sizes = config.get("dnn_sizes", [128, 64])
        input_dim = rnn_size

        for size in dnn_sizes:
            self.dnn_layers.append(nn.Linear(input_dim, size))
            self.dnn_layers.append(nn.ReLU())
            dropout_prob = 1.0 - config.get("dnn_keep_prob", 1.0)
            if dropout_prob > 0:
                self.dnn_layers.append(nn.Dropout(dropout_prob))
            input_dim = size

        self.output_layer = nn.Linear(input_dim, vocab_size)

    def forward(self, sym_seq, audio_seq, other_seq, state=None):
        # sym_seq: (batch, seq_len)
        # audio_seq: (batch, seq_len, channels, time, freq)
        # other_seq: (batch, seq_len, n_other)

        batch_size, seq_len = sym_seq.size()

        if self.embedding:
            x_sym = self.embedding(sym_seq)  # (batch, seq, embed_size)
        else:
            # Fallback
            x_sym = F.relu(self.sym_dense(sym_seq.unsqueeze(-1).float()))

        # Process Audio: Collapse batch and seq dims for CNN
        # audio_seq shape: (batch, seq, C, H, W)
        audio_reshaped = audio_seq.view(batch_size * seq_len, *audio_seq.shape[2:])

        x_audio = audio_reshaped
        for layer in self.cnn_layers:
            x_audio = layer(x_audio)

        x_audio = x_audio.reshape(x_audio.size(0), -1)  # Flatten

        if self.cnn_dim_reduction:
            x_audio = F.relu(self.cnn_dim_reduction(x_audio))

        x_audio = x_audio.view(batch_size, seq_len, -1)  # Restore sequence dim

        # Concatenate
        x = torch.cat([x_sym, x_audio, other_seq], dim=2)

        # RNN
        x, state = self.rnn(x, state)

        # DNN
        for layer in self.dnn_layers:
            x = layer(x)

        # Output
        logits = self.output_layer(x)
        return logits, state  # Return logits and state for autoregressive inference
