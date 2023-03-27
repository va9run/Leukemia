from Task_1.A_Libraries import *
from Task_1.C_DataImport import dataImport
from Task_2.A_data_preProcessing import load_images_labels_from_df
from Task_1.A1_path import *

def basicModel(trainPath,trainingPathType,validationPath,validationPathType,imageSize):
    training_generator,training_images = load_images_labels_from_df(trainPath,trainingPathType,imageSize)
    validation_generator,validation_images = load_images_labels_from_df(validationPath,validationPathType,imageSize)

    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu',input_shape=(imageSize-20, imageSize-20,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        training_generator,
        steps_per_epoch=training_images//32,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_images//32
    )

    loss, accuracy = model.evaluate(validation_generator,steps=validation_images//32)

    return f"Validation loss: {loss: .4f}", f"Validation accuracy: {accuracy: .4f}"


basicModel([trainingDataALL,trainingDataHEM],['ALL','HEM'],[validationDataALL,validationDataHEM],['ALL','HEM'],128)