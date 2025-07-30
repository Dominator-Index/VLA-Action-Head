from huggingface_hub import HfApi

api = HfApi()
info = api.model_info("gpt2")
print(info.modelId)
print(info.cardData)
print(info.tags)
print(info.pipeline_tag)