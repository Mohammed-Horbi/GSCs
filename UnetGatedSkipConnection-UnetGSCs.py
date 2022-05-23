# Unet Gated Skip Connection - Unet GSCs Model
class _ConvBlock(tf.keras.Model):

    def __init__(self, num_filters, name=None):
        super().__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        return x


class _EncoderBlock(tf.keras.Model):

    def __init__(self, num_filters, name=None):
        super().__init__(name=name)
        self.conv = _ConvBlock(num_filters)
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

    def call(self, inputs, training=None):
        
        x = self.conv(inputs, training=training)
        x_down_sampled = self.pool(x)
        return x_down_sampled, x


class _DecoderBlock(tf.keras.Model):

    def __init__(self, num_filters, use_dropout=False, name=None):
        super().__init__(name=name)

        self.dconv = tf.keras.layers.Conv2DTranspose(num_filters, 
                                                     (2, 2), 
                                                     strides=(2, 2), 
                                                     padding='same')
        if use_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)
        
        self.use_dropout = use_dropout

        self.bn = tf.keras.layers.BatchNormalization()
        self.conv1 = _ConvBlock(num_filters)
        self.conv2 = _ConvBlock(num_filters)
        self.conv11 = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same')
    
    def call(self, inputs, training=None):
        
        T2, T1 = inputs
        
        T2 = self.dconv(T2)
        
        T2 = self.bn(x, training=training)
        
        if self.use_dropout:
            T2 = self.dropout(x, training=training)
        
        T2 = tf.nn.relu(T2)
        I21 = tf.keras.layers.Concatenate()([T2, T1])
        
        A = self.conv11(I21)
        w = tf.nn.sigmoid(A)
        I3 = w * T1 + T2
        
        T2 = self.conv1(I3)
        T2 = self.conv2(T2)


        return T2        


class Encoder(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.enc1 = _EncoderBlock(32, name="enc1")
        self.enc2 = _EncoderBlock(64, name="enc2")
        self.enc3 = _EncoderBlock(128, name="enc3")
        self.enc4 = _EncoderBlock(256, name="enc4")
        self.enc5 = _EncoderBlock(512, name="enc5")

        self.conv = _ConvBlock(1024, name="center")

    def call(self, inputs, training=None):
        
        x1_down_sampled, x1 = self.enc1(inputs, training=training)
        
        x2_down_sampled, x2 = self.enc2(x1_down_sampled, training=training)
        x3_down_sampled, x3 = self.enc3(x2_down_sampled, training=training)
        x4_down_sampled, x4 = self.enc4(x3_down_sampled, training=training)
        x5_down_sampled, x5 = self.enc5(x4_down_sampled, training=training)

        x = self.conv(x5_down_sampled, training=training)

        return x, [x5, x4, x3, x2, x1]


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dec1 = _DecoderBlock(512, name="dec1", use_dropout=True)
        self.dec2 = _DecoderBlock(256, name="dec2", use_dropout=True)
        self.dec3 = _DecoderBlock(128, name="dec3")
        self.dec4 = _DecoderBlock(64, name="dec4")
        self.dec5 = _DecoderBlock(32, name="dec5")

    def call(self, inputs, training=None):
        x, [x5, x4, x3, x2, x1] = inputs

        x = self.dec1([x, x5], training=training)
        x = self.dec2([x, x4], training=training)
        x = self.dec3([x, x3], training=training)
        x = self.dec4([x, x2], training=training)
        x = self.dec5([x, x1], training=training)

        return x
    
class SemanticSegmentationModel(tf.keras.Model):

    def __init__(self, encoder, decoder, nb_classes, name=None):
        super().__init__(name=name)

        self.encoder = encoder
        self.decoder = decoder
        self.out_layer = tf.keras.layers.Conv2D(nb_classes, (1, 1), activation='softmax')        

    def call(self, inputs, training=None):

        enc_out = self.encoder(inputs, training=training)
        dec_out = self.decoder(enc_out, training=training)
        logits = self.out_layer(dec_out)

        return logits
