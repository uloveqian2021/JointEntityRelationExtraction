# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-1 上午8:54
@IDE    :PyCharm
@document   :layer_norm_ops.py
"""
import tensorflow as tf
""" 利用tensorflow 实现ConditionalLayerNorm　简化版本
　　参考　https://kexue.fm/archives/7124
"""


class ConditionalLayerNorm:
    def __init__(self,
                 normalized_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps
        self.beta = tf.Variable(tf.zeros(normalized_shape), name='beta')
        self.gamma = tf.Variable(tf.ones(normalized_shape), name='gamma')
        self.output_shape = normalized_shape
        self.beta_dense = tf.layers.Dense(units=normalized_shape, use_bias=False, kernel_initializer=tf.zeros_initializer)
        self.gamma_dense = tf.layers.Dense(units=normalized_shape, use_bias=False, kernel_initializer=tf.zeros_initializer)

    def __call__(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = tf.expand_dims(cond, 1)  # (b, 1, h*2) 扩展一个维度

        beta = self.beta_dense(cond) + self.beta  # (b, 1, h)
        gamma = self.gamma_dense(cond) + self.gamma  # (b, 1, h)

        mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = tf.reduce_mean(outputs ** 2, axis=-1, keep_dims=True)
        std = tf.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * gamma + beta

        return outputs

    @staticmethod
    def moments_for_ln(x, axes=1, name=None, epsilon=0.001):
        """output for mean and variance should be [batch_size]"""
        if not isinstance(axes, list):
            axes = list(axes)
        with tf.op_scope([x, axes], name, "moments"):
            # 均值
            mean = tf.reduce_mean(x, axes, keep_dims=True)
            # 方差
            variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
            return mean, variance

