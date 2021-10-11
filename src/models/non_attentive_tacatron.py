import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f

from src.models.config import (
    GaussianUpsampleConfig,
    TacatronConfig,
    TacatronEncoderConfig,
    TacatronDurationConfig,
    TacatronRangeConfig,
)
from src.models.layers import ConvNorm, LinearNorm, PositionalEncoding
from src.models.utils import norm_emb_layer


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x):
        for linear in self.layers:
            x = f.dropout(f.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    def __init__(self, config):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    config.n_mel_channels,
                    config.postnet_embedding_dim,
                    kernel_size=config.postnet_kernel_size,
                    stride=1,
                    padding=int((config.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(config.postnet_embedding_dim),
            )
        )

        for i in range(1, config.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        config.postnet_embedding_dim,
                        config.postnet_embedding_dim,
                        kernel_size=config.postnet_kernel_size,
                        stride=1,
                        padding=int((config.postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(config.postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    config.postnet_embedding_dim,
                    config.n_mel_channels,
                    kernel_size=config.postnet_kernel_size,
                    stride=1,
                    padding=int((config.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(config.n_mel_channels),
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = f.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = f.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class DurationPredictor(nn.Module):
    def __init__(self, embedding_dim, config: TacatronDurationConfig):
        super(DurationPredictor, self).__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = nn.Linear(config.lstm_hidden * 2, 1)

    def forward(self, x, input_lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = f.dropout(outputs, self.dropout, self.training)
        x = self.projection(outputs)
        return x


class RangePredictor(nn.Module):
    def __init__(self, embedding_dim, config: TacatronRangeConfig):
        super(RangePredictor, self).__init__()

        self.lstm = nn.LSTM(
            embedding_dim + 1,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = nn.Linear(config.lstm_hidden * 2, 1)

    def forward(self, x, durations, input_lengths):

        x = torch.cat((x, durations), dim=-1)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = f.dropout(outputs, self.dropout, self.training)
        x = self.projection(outputs)
        return x


class Attention(nn.Module):
    def __init__(self, embedding_dim, config: GaussianUpsampleConfig, device):
        super(Attention, self).__init__()
        self.duration_predictor = DurationPredictor(
            embedding_dim, config.duration_config
        )
        self.range_predictor = RangePredictor(embedding_dim, config.range_config)
        self.eps = config.eps
        self.dropout = config.attention_dropout
        self.positional_encoder = PositionalEncoding(
            embedding_dim, max_len=config.max_len, dropout=config.positional_dropout
        )
        self.device = device

    def calc_scores(self, durations, ranges):

        duration_cumsum = torch.cumsum(durations, dim=1).float()
        max_duration = duration_cumsum[:, -1, :].max()
        c = duration_cumsum - 0.5 * durations
        t = torch.arange(0, max_duration.item()).view(1, 1, -1).to(self.device)

        weights = torch.exp(-(ranges ** -2) * ((t - c) ** 2))
        weights_norm = torch.sum(weights, dim=1, keepdim=True) + self.eps
        weights = weights / weights_norm

        return weights

    def forward(self, embeddings, input_lengths):

        input_lengths = input_lengths.cpu().numpy()

        durations = self.duration_predictor(embeddings, input_lengths)
        ranges = self.range_predictor(embeddings, durations, input_lengths)

        scores = self.calc_scores(durations, ranges)

        attented_embeddings = torch.matmul(scores.transpose(1, 2), embeddings)
        attented_embeddings = self.positional_encoder(attented_embeddings)
        return durations, attented_embeddings


class Encoder(nn.Module):
    def __init__(self, phonem_embedding_dim, config: TacatronEncoderConfig):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(config.n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    phonem_embedding_dim,
                    phonem_embedding_dim,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=int((config.kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(phonem_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(
            phonem_embedding_dim,
            config.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, phonem_emb, input_lengths):
        for conv in self.convolutions:
            phonem_emb = f.dropout(f.relu(conv(phonem_emb)), 0.5, self.training)

        phonem_emb = phonem_emb.transpose(1, 2)
        phonem_emb = nn.utils.rnn.pack_padded_sequence(
            phonem_emb, input_lengths, batch_first=True
        )

        self.lstm.flatten_parameters()
        phonem_emb, _ = self.lstm(phonem_emb)

        phonem_emb, _ = nn.utils.rnn.pad_packed_sequence(phonem_emb, batch_first=True)

        return phonem_emb
    
    
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.n_mel_channels = config.n_mel_channels
        self.encoder_embedding_dim = config.encoder_embedding_dim
        self.attention_rnn_dim = config.attention_rnn_dim
        self.decoder_rnn_dim = config.decoder_rnn_dim
        self.prenet_dim = config.prenet_dim
        self.max_decoder_steps = config.max_decoder_steps
        self.gate_threshold = config.gate_threshold
        self.teacher_forcing_ratio = config.teacher_forsing_ratio
        self.p_attention_dropout = config.p_attention_dropout
        self.p_decoder_dropout = config.p_decoder_dropout

        self.prenet = Prenet(
            config.n_mel_channels,
            [config.prenet_dim, config.prenet_dim])

        self.decoder_rnn = nn.LSTM(
            config.attention_rnn_dim + config.encoder_embedding_dim,
            config.decoder_rnn_dim, num_layers=config.decoder_num_layers, bidirectional=False, batch_first=True)

        self.linear_projection = LinearNorm(
            config.decoder_rnn_dim + config.encoder_embedding_dim,
            config.n_mel_channels * config.n_frames_per_step)

    def forward(self, memory, ymels):

        decoder_input = Variable(torch.zeros(memory.size(0), 1, self.n_mel_channels))
        ymels = torch.cat((decoder_input, ymels[:, :-1, :]), dim=0)
        ymels = self.prenet(ymels)

        mel_outputs = []
        decoder_state = None
        decoder_input = torch.cat((ymels[:, 0, :], memory[:, 0, :]), dim=-1)
        for i in range(memory.size(1)):
            out, decoder_state = self.decoder_rnn(decoder_input.unsqueeze(1), decoder_state)
            out = torch.cat((out, memory[:, i, :].unsqueeze(1)), dim=-1)
            mel_out = self.linear_projection(out)
            mel_outputs.append(mel_out)
            if random.uniform(0, 1) > self.teacher_forcing_ratio:
                decoder_input = ymels[:, i, :]
            else:
                decoder_input = mel_out.squeeze(1)
            decoder_input = torch.cat((decoder_input, memory[:, i, :]), dim=-1)

        mel_outputs = torch.cat(mel_outputs, dim=1)
        return mel_outputs


class NonAttentiveTacatron(nn.Module):
    def __init__(self, config: TacatronConfig):
        super(NonAttentiveTacatron, self).__init__()
        self.phonem_embedding = nn.Embedding(
            config.n_phonems, config.phonem_embedding_dim
        )
        self.speaker_embedding = nn.Embedding(
            config.n_speakers, config.speaker_embedding_dim
        )
        norm_emb_layer(
            self.phonem_embedding, config.n_phonems, config.phonem_embedding_dim
        )
        norm_emb_layer(
            self.speaker_embedding, config.n_speakers, config.speaker_embedding_dim
        )
        self.encoder = Encoder(
            config.phonem_embedding_dim,
            config.encoder_config,
        )
        self.attention = Attention(
            config.phonem_embedding_dim + config.speaker_embedding_dim,
            config.attention_config,
            config.device,
        )
        self.decoder = Decoder(config)

    def forward(self, inputs):
        text_inputs, text_lengths, mels, speaker_ids = inputs

        phonem_emb = self.phonem_embedding(text_inputs).transpose(1, 2)
        speaker_emb = self.speaker_embedding(speaker_ids)

        phonem_emb = self.encoder(phonem_emb, text_lengths)

        speaker_emb = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, speaker_emb), dim=-1)

        durations, attented_embeddings = self.attention(embeddings, text_lengths)
        mel_outputs = self.decoder(attented_embeddings, mels)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return durations, mel_outputs_postnet


if __name__ == "__main__":
    rangec = TacatronRangeConfig()
    duration = TacatronDurationConfig()
    encoder = TacatronEncoderConfig()
    attention = GaussianUpsampleConfig(duration_config=duration, range_config=rangec)
    test_config = TacatronConfig(encoder_config=encoder, attention_config=attention)
    test_text_inputs = torch.ones(16, 24, dtype=torch.int64)
    test_text_lengts = torch.ones(16, dtype=torch.int64) * 24
    test_speaker_ids = torch.ones(16, 1, dtype=torch.int64)
    model = NonAttentiveTacatron(test_config)
    model((test_text_inputs, test_text_lengts, test_speaker_ids))
