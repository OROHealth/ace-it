from utils import *


def dblock(valid_pct, size1, size2, rotate, zoom):
    ''' Fetch images, labels.
        Splits to a train and validation set.
        Sets initial image size.
        Image transformations including resize, rotation, zooming and normalization.'''
    
    dblock = DataBlock(
        blocks=(ImageBlock(), CategoryBlock()),
        get_items=get_image_files,
        splitter=RandomSplitter(seed=42, valid_pct=valid_pct),
        get_y=parent_label,
        item_tfms=Resize(size1),
        batch_tfms=[(*aug_transforms(size=size2, max_rotate=rotate, max_zoom=zoom)),
                    Normalize.from_stats(*imagenet_stats)],)
    return dblock
