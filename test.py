import torch
from torchvision import utils as vutils
from models.LDRN import Generator
from PIL import Image
import torchvision.transforms as transforms
import os


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def main():
    net = Generator()
    net.cuda()
    dirs = 'RESULT'
    param = torch.load("./weight/LDRN_16_64.pth")
    net.load_state_dict(param, True)
    num = [5, 14, 100, 100]
    dataset = ['set5', 'set14', 'B100', 'Urban100']
    trans = transforms.Compose([transforms.ToTensor()])

    with torch.no_grad():
        net.eval()
        for j in range(4):
            if not os.path.exists(os.path.join('./datasets/test/' + dataset[j] + '/' + dirs)):
                os.makedirs(os.path.join('./datasets/test/' + dataset[j] + '/' + dirs))
            for i in range(num[j]):
                image = Image.open('./datasets/test/' + dataset[j] + '/' + 'test_lr/'+str(i+1)+".png").convert('RGB')
                image = trans(image).unsqueeze(0).cuda()
                print(image.shape)
                output = net(image)
                img_path = './datasets/test/' + dataset[j] + '/' + dirs + '/' + str(i+1) + '.png'

                save_image_tensor(output, img_path)


if __name__ == '__main__':
    main()
