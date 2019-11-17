import keras.backend as K
from keras.activations import tanh, softmax
from keras.engine import InputSpec
from keras.layers import LSTM, GRU
from keras.layers.recurrent import Recurrent


class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super(PointerLSTM, self).__init__(*args, **kwargs)

    def get_initial_states(self, x_input):
        return Recurrent.get_initial_state(self, x_input)

    def get_constants(self, x_input):
        return Recurrent.get_constants(self, x_input)

    def preprocess_input(self, x_input):
        return Recurrent.preprocess_input(self, x_input)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        # init = initializations.get('orthogonal')
        self.W1 = self.add_weight(name="W1",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.W2 = self.add_weight(name="W2",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.vt = self.add_weight(name="vt",
                                  shape=(input_shape[1], 1),
                                  initializer='uniform',
                                  trainable=True)
        self._trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1] - 1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = self.get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        # print "x_input:", x_input, x_input.shape
        # <TensorType(float32, matrix)>

        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).cell.call(x_input, states[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = _time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = _time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x
