from vilas import vilas_score
import utils
import timm
import torch
import torchattacks
import numpy as np


def report_vilas_scores(image_path, label, target_model):
	image = utils.get_image(image_path)

	# get clean image score
	score = vilas_score(image, target_model)

	# get attacked image score
	atk = torchattacks.PGD(target_model, eps=8 / 255, alpha=2 / 255, steps=4)
	label = torch.from_numpy(np.array([label]))
	image = torch.unsqueeze(image, dim=0)
	adv_image = atk(image, label)
	adv_image = torch.squeeze(adv_image)

	score_adv = vilas_score(adv_image, target_model)
	print(image_path)
	print('clean score: {}'.format(score))
	print('adv score: {}'.format(score_adv))
	print()

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=43, global_pool='avg')
model_dir = 'models/vit.pt'
model.load_state_dict(torch.load(model_dir))

images = ['images/00000.ppm', 'images/00001.ppm', 'images/00011.ppm', 'images/00035.ppm', 'images/00057.ppm']
labels = [16, 1, 7, 17, 26]

for i in range(5):
	report_vilas_scores(images[i], labels[i], model)






