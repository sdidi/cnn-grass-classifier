import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained MobileNetV2 model (without top layers)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tune the model by allowing the last few layers of MobileNetV2 to be trained
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Add custom layers on top of the pre-trained model for grass classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification: grass or not grass

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator for training data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to the range [0, 1]
    horizontal_flip=True,  # Randomly flip images horizontally
    rotation_range=30,  # Randomly rotate images by up to 30 degrees
    zoom_range=0.2,  # Random zoom
    shear_range=0.2  # Random shear transformation
)

# Load the training data from the 'grass' folder
train_generator = train_datagen.flow_from_directory(
    './grass',  # same directory
    target_size=(224, 224),  # Resize images to 224x224 (MobileNetV2 input size)
    class_mode='binary',  # Binary classification (grass vs non-grass)
    batch_size=32,  # Process 32 images per batch
    shuffle=True  
)

print(f"Number of training samples: {train_generator.samples}")
print(f"Classes: {train_generator.class_indices}")

# Use ImageDataGenerator for the test dataset (only rescale, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the test data from the 'grasstest' folder same properties as training
test_generator = test_datagen.flow_from_directory(
    './grasstest',  
    target_size=(224, 224), 
    class_mode='binary',  
    batch_size=32,  
    shuffle=False  
)

print(f"Number of test samples: {test_generator.samples}")

class_weights = {0: 1.0, 1: 5.0} 
history = model.fit(
    train_generator, 
    epochs=20,  # Number of epochs
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    class_weight=class_weights
)

# Plot the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f'Test accuracy: {test_acc:.2f}')


test_generator.reset()  # Reset the test generator for predictions
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)

# Flatten the predictions 
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Get true labels from the test generator
true_classes = test_generator.classes

#save the model
model.save('grass_classifier_model.keras')


