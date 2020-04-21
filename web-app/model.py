import torch
import torchvision.models
import torchvision.transforms as transforms
class model:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.resnet50()
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=1)
        self.model.load_state_dict(torch.load('model/model-resnet50.pth', map_location=self.device)) 
        self.model.eval().to(self.device)
    def prepare_image(self,image):
        Transform = transforms.Compose([
                #transforms.Resize([224,224]),      
                transforms.ToTensor(),
                ])
        image = Transform(image)
        image = image.unsqueeze(0)
        return image.to(self.device)

    def predict(self,image):
        print(self.device)
        with torch.no_grad():
            image = self.prepare_image(image)
            pred = str(self.model(image).item())[:4]
        return pred
