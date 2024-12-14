import torch
from transformers import DistilBertTokenizer
from vlm import VLM
import utils

def vilas_score(image, target_model):
	# model load
	device = 'cuda'
	torch.set_printoptions(sci_mode=False)

	target_model.eval()
	target_model = target_model.to(device)

	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
	vlm = VLM().to(device)
	vlm.load_state_dict(torch.load('models/vlm.pt', map_location=device))
	vlm.eval()

	#

	text_label_list = []
	l_map = utils.label_map
	for j in range(len(l_map)):
		text_label_list.append(l_map[str(j)].replace('_', ' '))

	text_embeddings_list = []
	for i in range(len(text_label_list)):
		encoded_query = tokenizer([text_label_list[i]])
		batch = {
			key: torch.tensor(values).to('cuda')
			for key, values in encoded_query.items()
		}
		with torch.no_grad():
			text_features = vlm.text_encoder(
				input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
			)
			text_embeddings = vlm.text_projection(text_features)
			text_embeddings_list.append(text_embeddings)
	text_embeddings_list = torch.stack(text_embeddings_list)
	text_embeddings_list = torch.squeeze(text_embeddings_list)

	image = image.to(device)
	image = image.to(torch.float32)

	image = torch.unsqueeze(image, dim=0)

	outputs = target_model(image)
	_, prediction_label = torch.max(outputs.data, 1)
	model_prediction_probs = outputs.softmax(dim=1)[0]

	# get image embedding
	image_features = vlm.image_encoder(image)
	image_embeddings = vlm.image_projection(image_features)

	# get similarity scores
	logits_per_image = image_embeddings @ text_embeddings_list.T

	vlm_probs = logits_per_image.softmax(dim=-1)
	vlm_probs = torch.squeeze(vlm_probs)

	kl1 = utils.kl_divergence(model_prediction_probs, vlm_probs)
	kl2 = utils.kl_divergence(vlm_probs, model_prediction_probs)
	kl_difference = (kl1 + kl2) / 2

	return kl_difference
