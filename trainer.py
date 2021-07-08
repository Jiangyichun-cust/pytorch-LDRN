import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from tools.ssim import msssim
from torch.autograd import Variable
from torchvision import transforms
import PIL.Image as Image


class Trainer(object):
    def __init__(self, net_g, optimizer_g, dataloader, device, batchsize, c=10.0):
        self.net_g = net_g
        self.optimizer_g = optimizer_g
        self.dataloader = dataloader
        self.device = device
        self.c = c
        self.batchsize = batchsize

    def train(self, epoch, writer):
        L1_criterion = nn.L1Loss()
        self.net_g.train()

        cnt = trange(118101 // self.batchsize + 1)
        for step, image in enumerate(self.dataloader):
            hr = Variable(image['hr']).to(self.device)
            lr = Variable(image['lr']).to(self.device)
            sr = self.net_g(lr)

            mseloss = L1_criterion(sr, hr)
            ssim = 1 - msssim(sr, hr).mean()
            loss_g = 0.9 * mseloss + 0.1 * ssim

            # loss_g = mseloss
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()
            cnt.set_description("[ Training upscaling! Epoch %i]" % (epoch + 1))
            cnt.set_postfix(L1_loss = mseloss.item(), mssim_loss = ssim.item(), total_loss=loss_g.item())
            cnt.update(1)
        del sr, loss_g, mseloss
        cnt.close()

        print(self.valid(writer=writer, epoch=epoch))
        return 0

    def valid(self, writer, epoch):
        with torch.no_grad():
            self.net_g.eval()
            trans = transforms.Compose([transforms.ToTensor()])
            cal_mse = nn.MSELoss()
            total_mse = 0.
            total_ssim = 0.
            for i in range(100):
                lr = Image.open('./datasets/B100_vaild_LR/' + str(i+1) +'.png')
                hr = Image.open('./datasets/B100_vaild_HR/' + str(i+1) +'.png')
                lr = trans(lr).unsqueeze(0).cuda()
                hr = trans(hr).unsqueeze(0).cuda()
                sr = self.net_g(lr)
                if epoch == 0:
                    writer.add_image("Original HR-epoch" + str(epoch), hr.squeeze(0), global_step=i, walltime=None, dataformats='CHW')
                    writer.add_image("Original LR-epoch" + str(epoch), lr.squeeze(0), global_step=i, walltime=None, dataformats='CHW')
                writer.add_image("after-epoch" + str(epoch), sr.squeeze(0), global_step=i, walltime=None, dataformats='CHW')
                mse = cal_mse(sr, hr).item()
                ssim = 1 - msssim(sr, hr).mean().item()
                total_mse = total_mse + mse
                total_ssim = total_ssim + ssim


            writer.add_scalar('valid MSE Loss', total_mse / 100, global_step=epoch)
            writer.add_scalar('valid SSIM', total_ssim / 100, global_step=epoch)
            print("mean MSE = %f, mean SSIM = %f"%(total_mse/100, total_ssim/100))
            del hr, lr, sr, total_mse, total_ssim, mse, ssim
            torch.save(self.net_g.state_dict(), './weight/no_mul/last.pth')
            return "*" * 160