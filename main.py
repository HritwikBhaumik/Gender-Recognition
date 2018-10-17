from fastai.conv_learner import *
from fastai.dataset import *
from fastai.plots import *
import os


sz = 224
arch = resnet34
PATH = os.path.join(os.getcwd(),'manandwomen')


tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
print(data.classes)
learn = ConvLearner.pretrained(arch, data, precompute=True)
print('loading requirements......')
# print('This has been made by shaaran alias devshaaran, if you are using this code anywhere for research or educational purposes, please give reference.ENJOY!')
learn.precompute=False
#learn.fit(1e-1, 1)
learn.fit(1e-1, 3, cycle_len=1)
learn.load('224_all')
print('loading done !')
trn_tfms, val_tfms = tfms_from_model(arch, sz)
im = val_tfms(open_image(g))
learn.precompute = False
preds = learn.predict_array(im[None])
print(data.classes[np.argmax(preds)])
