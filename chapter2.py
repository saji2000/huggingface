from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")

model.save_pretrained("bert-model")