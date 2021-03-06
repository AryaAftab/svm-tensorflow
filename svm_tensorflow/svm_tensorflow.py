import tensorflow as tf
from tensorflow.keras import layers



#classes

class LinearSVC(layers.Layer):
    def __init__(self, num_classes=2, **kwargs):
        super(LinearSVC, self).__init__(**kwargs)
        self.num_classes = num_classes
    
        self.reg_loss = lambda weight : 0.5 * tf.reduce_sum(tf.square(weight))

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.num_classes,), initializer=tf.constant_initializer(value=0.1),
            trainable=True
        )

    def call(self, inputs):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.reg_loss(self.w)
        self.add_loss(loss)
        
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(LinearSVC, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config




class SVMTrainer(tf.keras.Model):
    def __init__(
        self,
        num_class,
        C=1.0,
        bone=None,
        name="SVMTrainer",
        **kwargs
    ):
        super(SVMTrainer, self).__init__(name=name, **kwargs)
    
        self.num_class = num_class

        if bone is None:
            self.bone = lambda x: tf.identity(x)
        else:
            self.bone = bone

        self.linear_svc = LinearSVC(self.num_class)
        self.C = C
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    
    def svc_loss(self, y_true, y_pred, sample_weight, reg_loss):
        
        loss = tf.keras.losses.squared_hinge(y_true ,y_pred)
        if sample_weight is not None:
            loss = sample_weight * loss
        
        return reg_loss + self.C * loss
    
    
    def compile(self, **kwargs):
        super(SVMTrainer, self).compile(**kwargs)
        self.compiled_loss = None
    
    
    def call(self, x, training=False):
        x = self.bone(x)
        x = self.linear_svc(x)
        return x

    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.svc_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                reg_loss=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    
    def test_step(self, data):
        # Unpack the data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.svc_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                reg_loss=self.losses,
        )
        # Update the metrics.
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch or at the start of `evaluate()`.
        return [self.loss_tracker] + self.compiled_metrics.metrics
