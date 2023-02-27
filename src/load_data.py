from PIL import Image, ImageOps


def view_sample(subpath):
    '''View a sample image given the subpath.

    Where path = Path()
    Where the subpath = path/filename'''

    samp = Image.open(subpath)
    show_image(samp)


def verify_images(sets):
    '''Remove images that cannot be opened'''
    
    path = Path('/content/ace-it/data_3class_skin_diseases/')

    for s in sets:
        sub_path = path/s
        no_open = verify_images(sub_path.ls())
        print(no_open)
        no_open.map(Path.unlink)
