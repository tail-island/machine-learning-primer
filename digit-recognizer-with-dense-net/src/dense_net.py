import tensorflow as tf

from funcy import rcompose


def dense_net(growth_rate, classes):
    # KerasやTensorflowのレイヤーや関数をラップします。

    def average_pooling_2d(pool_size, strides=None):
        return tf.keras.layers.AveragePooling2D(pool_size, strides=strides)

    def concatenate():
        return tf.keras.layers.Concatenate()

    def batch_normalization():
        return tf.keras.layers.BatchNormalization(epsilon=1.001e-5)

    def conv_2d(filters, kernel_size):
        return tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)

    def dense(units):
        return tf.keras.layers.Dense(units)

    def global_average_pooling_2d():
        return tf.keras.layers.GlobalAveragePooling2D()

    def relu():
        return tf.keras.layers.ReLU()

    def softmax():
        return tf.keras.layers.Softmax()

    def zero_padding2d(padding):
        return tf.keras.layers.ZeroPadding2D(padding=padding)

    # DenseNetに必要な演算を定義します。

    def dense_block(blocks):
        def op(inputs):
            result = inputs

            for _ in range(blocks):
                result_ = batch_normalization()(result)
                result_ = relu()(result_)
                result_ = conv_2d(4 * growth_rate, 1)(result_)
                result_ = batch_normalization()(result_)
                result_ = relu()(result_)
                result_ = conv_2d(growth_rate, 3)(result_)

                result = concatenate()((result, result_))

            return result

        return op

    def transition_block():
        def op(inputs):
            result = batch_normalization()(inputs)
            result = relu()(result)
            result = conv_2d(int(tf.keras.backend.int_shape(inputs)[3] * 0.5), 1)(result)
            result = average_pooling_2d(2, strides=2)(result)

            return result

        return op

    def dense_net_121():
        return rcompose(dense_block(6),
                        transition_block(),
                        dense_block(12),
                        transition_block(),
                        dense_block(24),
                        transition_block(),
                        dense_block(16))

    return rcompose(conv_2d(64, 1),
                    batch_normalization(),
                    relu(),  # ImageNetの場合はここでMaxPooling2Dして画素数を減らすのですけど、今回はもともとの画素数が少ないのでやりません。
                    dense_net_121(),
                    batch_normalization(),
                    relu(),
                    global_average_pooling_2d(),
                    dense(classes),
                    softmax())
