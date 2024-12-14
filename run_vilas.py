from vilas import vilas_score
import utils
import timm
import torch
import torchattacks
import numpy as np


target_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=43, global_pool='avg')
model_dir = 'models/vit.pt'
target_model.load_state_dict(torch.load(model_dir))

image = utils.get_image('images/00011.ppm')
label = 7


# get clean image score
score = vilas_score(image, target_model)

# get attacked image score
atk = torchattacks.PGD(target_model, eps=8/255, alpha=2/255, steps=4)
label = torch.from_numpy(np.array([label]))
image = torch.unsqueeze(image, dim=0)
adv_image = atk(image, label)
adv_image = torch.squeeze(adv_image)

score_adv = vilas_score(adv_image, target_model)

print('clean score: {}'.format(score))
print('adv score: {}'.format(score_adv))



