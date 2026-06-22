"""
Load the DDQN "storm" weights (best weights, beta = 0.999).
Architecture: Input(50) -> Dense(300) -> Dense(300) -> Dense(11)

Problem:
    The .h5 file was saved with Keras 3. Loading it with Keras 2 raises
    "Layer count mismatch ... found 0 saved layers", because the two HDF5
    formats are incompatible.

Solution:
    Read the weights from the .h5 with h5py and assign them manually via
    set_weights(). This works with ANY Keras version and uses your original
    .h5 file directly.
"""

import h5py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    """Build the exact architecture matching the saved weights."""
    return keras.Sequential([
        keras.Input(shape=(50,)),
        layers.Dense(300, activation="relu"),
        layers.Dense(300, activation="relu"),
        layers.Dense(11, activation="linear"),  # 11 Q-values
    ])


def load_keras3_weights_into(model, h5_path):
    """Load Keras-3 weights into `model`, mapping layers IN ORDER.

    Drop-in replacement: swap `model.load_weights(PATH)` for
    `load_keras3_weights_into(model, PATH)`.
    """
    f = h5py.File(h5_path, "r")
    base = "_layer_checkpoint_dependencies"
    # Sorted as dense, dense_1, dense_2
    saved = sorted(f[base].keys(), key=lambda n: (len(n), n))

    target = [l for l in model.layers if l.weights]  # only layers that have weights
    if len(target) != len(saved):
        raise ValueError(f"{len(target)} layers in model vs {len(saved)} saved")

    for layer, sname in zip(target, saved):
        grp = f[f"{base}/{sname}/vars"]
        # vars/0 = kernel, vars/1 = bias
        w = [grp[k][:] for k in sorted(grp.keys(), key=int)]
        layer.set_weights(w)
    f.close()
    return model


if __name__ == "__main__":
    PATH = "storm_ddqn_best_weights_beta_0_999.h5"   # <-- your .h5 file
    model = build_model()
    load_keras3_weights_into(model, PATH)
    model.summary()

    q = model.predict(np.random.rand(1, 50).astype("float32"), verbose=0)
    print("Example Q-values (11 actions):", q[0])
