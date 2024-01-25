
import torch
from open_clip import create_model_from_pretrained, get_tokenizer 

class OpenClip(torch.nn.Module):
    def __init__(self, model_name_or_path, tokenizer_name=None):
        super().__init__()
        self.base, preprocess = create_model_from_pretrained(model_name_or_path)
        self.process_images = ImageProcessor(preprocess).process_images
        self.projection = self.base.visual.proj
        self.base.visual.proj = None
        if(tokenizer_name):
            self.process_text = TextProcessor(tokenizer_name, self.base.context_length).process_text
        else:
            del self.base.transformer # delete text encoder

    def forward(self, inputs):
        pooled_output = self.base.visual(inputs)
        projected_embedding = pooled_output @ self.projection
        return pooled_output, projected_embedding

    def encode_text(self, texts):
        return self.base.encode_text(texts)        

class ImageProcessor:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def process_images(self, pil_images):
        pil_images = pil_images if isinstance(pil_images, list) else [pil_images]
        processed = [self.preprocess(image) for image in pil_images]
        return torch.stack(processed)

class TextProcessor:
    def __init__(self, tokenizer_name, context_length):
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.context_length = context_length

    def process_text(self, text):
        text = text if isinstance(text, list) else [text]
        return self.tokenizer(text, context_length=self.context_length)

if __name__ == "__main__":
    from PIL import Image
    import torch.nn.functional as F
    labels = ["Person", "Plant"]

    model = OpenClip('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384', 'ViT-H-14')
    model = model.cuda()    
    image = model.process_images(Image.open("dudes.png")).cuda()
    text = model.process_text(labels).cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, v1 = model(image)
        t1 = model.encode_text(text)


    model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
    model = model.cuda()
    tokenizer = get_tokenizer('ViT-H-14')

    image = preprocess(Image.open("dudes.png")).unsqueeze(0).cuda()
    text = tokenizer(labels, context_length=model.context_length).cuda()

    with torch.no_grad(), torch.cuda.amp.autocast():
        v2 = model.encode_image(image)
        t2 = model.encode_text(text)

    # compare our model with the default output
    assert torch.allclose(v1, v2, atol=1e-4)
    assert torch.allclose(t1, t2, atol=1e-4)

    probs = F.normalize(v1, dim=-1) @ F.normalize(t1, dim=-1).T
    zipped_list = list(zip(labels, [round(p.item(), 3) for p in probs[0]]))
    print("Label probabilities: ", zipped_list)
