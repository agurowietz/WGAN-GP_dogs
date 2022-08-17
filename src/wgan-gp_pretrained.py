
import torch
from torch import nn
from torchvision import transforms
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import os

torch.manual_seed(999) 


device = "cuda" if torch.cuda.is_available() else "cpu"
workers = 4
lr = 0.0001
beta_1 = 0.0
beta_2 = 0.9
lambda_gp = 10
crit_iters = 5
n_epochs = 1000
img_channels = 3
z_dim = 64
hidden_dim = 64
batch_size = 64
image_size = 64
display_step = 20


dataset = dset.ImageFolder(root="data/chihuahua",
                           transform=transforms.Compose([
                               transforms.Resize((image_size, image_size)),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, pin_memory = True)



# Plot some training images
#real_batch = next(iter(dataloader))
#plt.figure(figsize=(8,8))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.show()


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, hidden_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 16, kernel_size=4, stride=1, padding = 0),
            self.make_gen_block(hidden_dim * 16, hidden_dim * 8, kernel_size=4, stride=2, padding = 1),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding =  1),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding =  1),
            self.make_gen_block(hidden_dim * 2, img_channels, kernel_size=4, stride=2, padding = 1, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size, stride, padding, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


class Critic(nn.Module):
    def __init__(self, img_channels, hidden_dim):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(img_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            self.make_crit_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            self.make_crit_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            self.make_crit_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            self.make_crit_block(hidden_dim * 8, 1, kernel_size=4, stride=2, padding=0, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size, stride, padding, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.InstanceNorm2d(output_channels, affine = True),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


def get_noise(batch_size, z_dim, device=device):
    return torch.randn(batch_size, z_dim, device=device)


def get_gradient(crit, real, fake, epsilon):
    # Interpolate images 
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    # Flatten the gradients
    gradient = gradient.view(len(gradient), -1)
    # Calculate gradient norm (magnitude of gradient)
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, lambda_gp):
    crit_loss = (-(torch.mean(crit_real_pred) - torch.mean(crit_fake_pred)) + lambda_gp * gp)
    return crit_loss


#def weights_init(m):
#    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#        torch.nn.init.normal_(m.weight, 0.0, 0.02)
#    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
#        torch.nn.init.normal_(m.weight, 0.0, 0.02)
#        torch.nn.init.constant_(m.bias, 0)
#gen = gen.apply(weights_init)
#crit = crit.apply(weights_init)


gen = Generator(z_dim, img_channels, hidden_dim).to(device)
gen.load_state_dict(torch.load("model/tsinghua_dogs/generator_iter_23.pth"))
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic(img_channels, hidden_dim).to(device) 
crit.load_state_dict(torch.load("model/tsinghua_dogs/critic_iter_23.pth"))
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

torch.backends.cudnn.benchmark = True

cur_step = 0
generator_losses = []
critic_losses = []
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    for batch_idx, (real, _) in enumerate(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        mean_iteration_critic_loss = 0
        for _ in range(crit_iters):
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, lambda_gp)

            mean_iteration_critic_loss += crit_loss.item() / crit_iters
            crit_loss.backward(retain_graph=True)
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)        
        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()
        gen_opt.step()
        generator_losses += [gen_loss.item()]

        #Visualization and saving progress every x epochs
        if epoch % 10 == 0 and batch_idx == 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            vutils.save_image(real, os.path.join("output", "real_samples.png"), normalize=True)
            vutils.save_image(fake.detach(), os.path.join("output", f"fake_samples_{epoch}.png"), normalize=True) 
            torch.save(gen.state_dict(), f"model/generator_iter_{epoch}.pth")
            torch.save(crit.state_dict(), f"model/critic_iter_{epoch}.pth")
            
            num_examples = (len(generator_losses))
           
            #Loss plot
            plt.plot(
                range(num_examples), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, 1).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, 1).mean(1),
                label="Critic Loss"
            )
            plt.ylim([-100, 50])
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
            plt.savefig(os.path.join("output", f"loss_detail_{epoch}.png"))
            plt.clf()
        
        cur_step += 1
