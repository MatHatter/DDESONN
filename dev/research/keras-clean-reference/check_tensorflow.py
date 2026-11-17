"""Generate independent TensorFlow gradient/update/state checks for the native adapter."""
import json
from pathlib import Path
from reference import build, parameters, moving, tensor, np, tf
root = Path('/workspace/keras-clean-100')
checks = []
for average in (False, True):
    model = build(17, 'mean' if average else 'sum')
    rng = np.random.default_rng(809)
    cases = []
    initial = dict(parameters=parameters(model), moving=moving(model),
                   accumulators=[tensor(np.full(v.shape, .1, dtype=np.float32)) for v in model.trainable_variables])
    for n in (16, 12):
        x = rng.normal(size=(n, 12)).astype(np.float32)
        y = rng.integers(0, 2, size=(n, 1)).astype(np.float32)
        mask = (rng.random((n, 64)) >= .1).astype(np.float32) / np.float32(.9)
        with tf.GradientTape() as tape:
            value = tf.convert_to_tensor(x)
            value = model.layers[0](value)
            value = model.layers[1](value, training=True)
            value = value * mask
            value = model.layers[3](value)
            value = model.layers[4](value, training=True)
            logits = tf.matmul(value, model.layers[6].kernel) + model.layers[6].bias
            probability = tf.math.sigmoid(logits)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            loss = (tf.reduce_mean(bce) if average else tf.reduce_sum(bce)) + tf.add_n([
                tf.reduce_sum(tf.abs(layer.kernel)) * tf.constant(.00028, tf.float32)
                for layer in (model.layers[0], model.layers[3], model.layers[6])])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        cases.append(dict(x=tensor(x), y=tensor(y), mask=tensor(mask), loss=float(loss.numpy()),
                          probability=tensor(probability.numpy()), gradients=[tensor(v.numpy()) for v in gradients],
                          parameters=parameters(model), moving=moving(model),
                          accumulators=[tensor(v.numpy()) for v in model.optimizer._accumulators]))
    inference = model.predict(np.tile(x, (7, 1)), batch_size=32, verbose=0)
    checks.append(dict(average=average, initial=initial, cases=cases,
                       inference_x=tensor(np.tile(x, (7, 1))), inference=tensor(inference)))
(root / 'tensorflow-checks.json').write_text(json.dumps(checks))
print('Generated two reductions, two optimizer steps, BN state and inference fixtures')
