import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
ResNet50 = tf.keras.applications.ResNet50
Sequential = tf.keras.models.Sequential
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dense = tf.keras.layers.Dense
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint


IMG_SIZE = (224, 224)
BATCH_SIZE = 8

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

train_gen = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)
val_gen = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE,3))
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("deepfake_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=[checkpoint])
