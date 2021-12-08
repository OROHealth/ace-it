from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')



def aug(x,re):

    i = 0
    for batch in datagen.flow(x,batch_size=2,
                              save_to_dir='data_3class_skin_diseases/data_aug/lichen_planus/', save_prefix='lichen_planus_{}_{}'.format(re,i), save_format='jpeg',seed=1):
        #save_prefix='cat_{}'.format(i)
        i += 1
        if i > 9:
            break  # otherwise the generator would loop indefinitely

arr = os.listdir('data_3class_skin_diseases/lichen_planus/')
for im in arr:

    re=im[:-4]
    print(re)
    img = load_img('data_3class_skin_diseases/lichen_planus/{file}'.format(file=im))  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # print(x)

    aug(x,re)