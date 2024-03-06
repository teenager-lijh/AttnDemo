from torchvision import transforms


# 你可以定义你自己的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])