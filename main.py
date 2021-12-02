from src.utils import *
from src.load_data import *
from src.transforms import *
from src.model import *

unzip \* zip & & rm*.zip

path = Path('./data_3class_skin_diseases')

view_sample(path/'acne/4-100.jpg')

sets = ['acne', 'herpes_simplex', 'lichen_planus']
verify_images(sets)

dblock(valid_pct=0.2, size1=460, size2=200, rotate=20, zoom=1.2)

dls = dblock.dataloaders(path, bs=64)

wandb.login()
wandb.init(project='Derm', name='simple_cnn')

learn = Learner(dls, simple_cnn, metrics=accuracy,
                cbs=[WandbCallback(), SaveModelCallback()])

lr_valley = learn.lr_find()

learn.fit_one_cycle(n_epoch=50, lr_max=lr_valley[0])

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

learn.export()

path_ = Path()
path_.ls(file_exts='.pkl')
