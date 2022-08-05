import clip
from PIL.Image import Image
import torch

class ClipModel:
    def __init__(self, model_name: str = 'RN50') -> None:
        """
        Available models
        ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32',
         'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        """
        self.model, self.img_preprocess = clip.load(model_name)
    
    def predict(self, images: list[Image], prompts: list[str]) -> dict:
        if len(images) == 1:
            return self.compute_prompts_probabilities(images[0], prompts)
        elif len(prompts) == 1:
            return self.compute_images_probabilities(images, prompts[0])
        else:
            raise ValueError('Either images or prompts must be a single element')
    
    def compute_prompts_probabilities(self, image: Image, prompts: list[str]) -> dict[str, float]:
        preprocessed_image = self.img_preprocess(image).unsqueeze(0)
        tokenized_prompts = clip.tokenize(prompts)
        with torch.inference_mode():
            image_features = self.model.encode_image(preprocessed_image)
            text_features = self.model.encode_text(tokenized_prompts)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            probs = list(logits_per_image.softmax(dim=-1).cpu().numpy()[0])

        scored_prompts = {tag: float(prob) for tag, prob in zip(prompts, probs)}
        return scored_prompts
    
    def compute_images_probabilities(self, images: list[Image], prompt: str) -> dict[Image, float]:
        raise
        preprocessed_images = [self.img_preprocess(image).unsqueeze(0) for image in images]
        tokenized_prompts = clip.tokenize(prompt)
        with torch.inference_mode():
            image_features = self.model.encode_image(preprocessed_image)
            text_features = self.model.encode_text(tokenized_prompts)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            probs = list(logits_per_image.softmax(dim=-1).cpu().numpy()[0])

        scored_prompts = {tag: float(prob) for tag, prob in zip(prompts, probs)}
        return scored_prompts