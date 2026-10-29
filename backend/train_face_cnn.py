import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ FINAL CORRECT DATASET PATH
DATASET_PATH = "D:/ai_voice_mental_health/dataset/fer2013/archive"

train_dir = f"{DATASET_PATH}/train"
test_dir = f"{DATASET_PATH}/test"

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(96, 96),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(96, 96),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32
)

base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(7, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3)
]

model.summary()

model.fit(
    train_data,
    validation_data=test_data,
    epochs=30,
    callbacks=callbacks
)

model.save("emotion_cnn.h5")

print("🔥 80%-level FER2013 model trained & saved successfully!")
