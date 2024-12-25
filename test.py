import torch
from PIL import Image
from torchvision import transforms
from training import MojaMreza, MojaMrezaOG
import matplotlib.pyplot as plt
import json

def load_labels():
    with open('labels.json', 'r') as openfile:
        json_object = json.load(openfile)
        labels_array = [json_object[key] for key in sorted(json_object, key=int)]
        return labels_array

def eval_image(img_path):
    image = Image.open(img_path).convert("RGB")  # Ensure 3 channels (RGB)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)  # Raw logits
        predicted_class = torch.argmax(output, dim=1).item()
        print("Predicted: ", categories[predicted_class])
        print(torch.nn.Softmax(dim=1)(output))
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze(0).numpy()

    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    fig.suptitle(f'Predicted: {categories[predicted_class]}', fontsize=16)
    axarr[0].imshow(image)
    axarr[0].axis("off")

    axarr[1].barh(categories, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

categories = load_labels()
num_classes = len(categories)

#model = MojaMreza(num_classes)
#model.load_state_dict(torch.load("model_0_timm.pth", map_location=torch.device('cpu')))
model = MojaMrezaOG(num_classes)
model.load_state_dict(torch.load("model_0.pth", map_location=torch.device('cpu')))
model.eval()
print("Model loaded successfully.")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

eval_image("test_images/clear0.jpg")
eval_image("test_images/clear1.jpg")
eval_image("test_images/clear2.jpg")
eval_image("test_images/cloudy0.jpeg")
eval_image("test_images/rainy0.jpg")
eval_image("test_images/rainy1.jpg")
eval_image("test_images/rainy2.jpg")