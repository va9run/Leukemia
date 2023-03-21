from Task_1.A_Libraries import *
from Task_1.C_DataImport import dataImport

def load_images_labels_from_df(df, img_size):
    images = []
    labels = []
    
    for _, row in df.iterrows():
        label = row['Labels']
        img_data = row['Images']
        img_resized = img_data.resize((img_size, img_size)) #resize images (128*128, 256*256 etc)
        img_array = np.array(img_resized) / 255.0  # Normalize pixel values
        images.append(img_array)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    # Augmentation: A technique to create more and different images from existing images using various transformation method
    augmentation = preprocessing.image.ImageDataGenerator(
                                                        rotation_range=20, # image rotation range
                                                        width_shift_range=20, # horizontal shift during augmentation
                                                        height_shift_range=20, # vertical shift during augmentation
                                                        shear_range=20, # shear angle shift during augmentation
                                                        zoom_range=20, # image zoom-in or out during augmentation
                                                        horizontal_flip=True, 
                                                        fill_mode='nearest', # how to fill empty spaces created during augmentation*
                                                        vertical_flip=True
                                                    )
    
    augmentation.fit(images)
    augmentedImage_generator = augmentation.flow(images, labels, batch_size=32)

    return augmentedImage_generator


# different fill_mode: 1) nearest 2)constant 3)reflect 4)wrap