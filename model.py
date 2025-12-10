import tensorflow as tf
from tensorflow.keras import layers, models, applications, Input

@tf.keras.utils.register_keras_serializable()
class Encoder(layers.Layer):
    def __init__(self, latent_dim=1024, freeze_backbone=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.freeze_backbone = freeze_backbone
        self.base_model = applications.EfficientNetB0(
            include_top=False, 
            weights='imagenet', 
            input_shape=(224, 224, 3)
        )
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_dim)
        self.leaky_relu = layers.LeakyReLU(negative_slope=0.1)

        if freeze_backbone:
            self.base_model.trainable = False
            print("🔒 Encoder Backbone: FROZEN")
        else:
            self.base_model.trainable = True
            print("🔓 Encoder Backbone: TRAINABLE")
    
        for layer in self.base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    def build(self, input_shape):        
        flatten_output_shape = (input_shape[0], 62720)
        self.dense.build(flatten_output_shape)        
        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.leaky_relu(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.latent_dim)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "freeze_backbone": self.freeze_backbone
        })
        return config

    def build_from_config(self, config):
        self.build((None, 224, 224, 3))

@tf.keras.utils.register_keras_serializable()
class DecoderBlock(models.Model):
    def __init__(self, filters, kernel_size=(3, 3, 2), is_residual=True, num_convs=2, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.is_residual = is_residual
        self.num_convs = num_convs
        
        self.upsample = layers.UpSampling3D(size=(2, 2, 2))
        
        self.convs = []
        for _ in range(num_convs):
            self.convs.append(
                layers.Conv3D(filters, kernel_size, padding='same', activation='relu')
            )
            
        if self.is_residual:
            self.final_relu = layers.Activation('relu')
            self.project_residual = layers.Conv3D(filters, (1, 1, 1), padding='same', activation=None)
            
    def build(self, input_shape):
        super(DecoderBlock, self).build(input_shape)
        
    def call(self, inputs):
        x = self.upsample(inputs)
        residual_path = x
        
        for conv in self.convs:
            x = conv(x)
            
        if self.is_residual:
            if x.shape[-1] != residual_path.shape[-1]:
                residual_path = self.project_residual(residual_path)
            x = layers.add([x, residual_path])
            x = self.final_relu(x)
        return x

    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "is_residual": self.is_residual,
            "num_convs": self.num_convs
        })
        return config

@tf.keras.utils.register_keras_serializable()
class R2N2Decoder(layers.Layer):
    def __init__(self, filters=128, **kwargs):
        super(R2N2Decoder, self).__init__(**kwargs)
        self.filters = filters
        k_size = (3, 3, 3)
        self.block1 = DecoderBlock(filters, k_size, is_residual=False, num_convs=3)
        self.block2 = DecoderBlock(filters // 2, k_size, num_convs=3)
        self.block3 = DecoderBlock(filters // 4, k_size, num_convs=2)
        self.block4 = DecoderBlock(filters // 8, k_size, num_convs=2)
        self.block5 = DecoderBlock(filters // 16, k_size, num_convs=2)
        self.final_conv = layers.Conv3D(1, k_size, padding='same', activation=None)
    def build(self, input_shape):
        super(R2N2Decoder, self).build(input_shape)
    
    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.final_conv(x)

    def get_config(self):
        config = super(R2N2Decoder, self).get_config()
        config.update({
            "filters": self.filters
        })
        return config

def build_model(seq_length=4, voxel_res=128, freeze_encoder=False):
    input_seq = Input(shape=(seq_length, 224, 224, 3))
    encoder = Encoder(latent_dim=1024, freeze_backbone=freeze_encoder)
    encoded_seq = layers.TimeDistributed(encoder)(input_seq)
    grid_size = 4
    grid_channels = 16
    reshaped_seq = layers.TimeDistributed(
        layers.Reshape((grid_size, grid_size, grid_size, grid_channels))
    )(encoded_seq)
    rnn_out = layers.ConvLSTM3D(
        filters=grid_channels,
        kernel_size=(3, 3, 3),
        padding='same',
        return_sequences=False,
        activation='tanh'
    )(reshaped_seq)
    output = R2N2Decoder(filters=128)(rnn_out)
    model = models.Model(inputs=input_seq, outputs=output, name="3D_Reconstruction_Net")
    return model