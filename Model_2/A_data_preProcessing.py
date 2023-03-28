from Model_1.A_Libraries import *
from Model_1.A1_path import *
from Model_1.C_DataImport import dataImport

def load_images_labels_from_df(path, pathType, img_size):

    df = dataImport(path,pathType)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        label = row['Labels']
        img_data = row['Images']
        img_resized = img_data.resize((img_size, img_size)) #resize images (128*128, 256*256 etc)
        img_array = exposure.equalize_adapthist(np.array(img_resized),clip_limit=0.03,nbins=256)  # Normalize pixel values
        img_array = (img_array*255).astype(np.uint8)
        images.append(img_array)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    # crop images
    cropped_images = []

    for i in images:
        cropped_image = i[10:-10, 10:-10, :] 
        cropped_images.append(cropped_image)

    cropped_images = np.array(cropped_images)

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
    
    augmentation.fit(cropped_images)
    augmentedImage_generator = augmentation.flow(cropped_images, labels, batch_size=32)


    return augmentedImage_generator, len(cropped_images)