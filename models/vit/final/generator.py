import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# ------------------------- Config -------------------------
img_size = 256
nz = 100
num_classes = 2
embedding_dim = 1  # for channel-wise embedding map
nc = 3
ngf = 64
ndf = 64
batch_size = 64
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
minority_class = 1  # <-- adjust as needed

# ------------------------- Data -------------------------
data_root = "Aerial_Landscapes"
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root=data_root, transform=transform)
minority_indices = [i for i, (_, label) in enumerate(dataset) if label == minority_class]
minority_subset = torch.utils.data.Subset(dataset, minority_indices)
minority_loader = DataLoader(minority_subset, batch_size=batch_size, shuffle=True)

# ------------------------- Models -------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.input = nn.Sequential(
            nn.Linear(nz + embedding_dim, ngf * 16 * 4 * 4),
            nn.BatchNorm1d(ngf * 16 * 4 * 4),
            nn.ReLU(True)
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], dim=1)
        x = self.input(x).view(-1, ngf * 16, 4, 4)
        return self.upconv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(nc + 1, ndf, 4, 2, 1),  # 256 → 128
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),  # 128 → 64
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),  # 64 → 32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),  # 32 → 16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, 1, 4, 2, 1),  # 16 → 8
        )
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,1,8,8] → [B,1,1,1]
            nn.Flatten(),             # [B,1,1,1] → [B]
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # [B, 1, 1, 1]
        label_map = label_map.expand(-1, 1, img.size(2), img.size(3))  # ✅ 保证跟图像同高宽
        x = torch.cat([img, label_map], dim=1)  # [B, 4, H, W]
        return self.final(self.conv(x)).view(-1)

# ------------------------- Train -------------------------
G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

os.makedirs("generated_minority", exist_ok=True)

for epoch in range(num_epochs):
    for i, (real_img, real_label) in enumerate(minority_loader):
        real_img = real_img.to(device)
        real_label = real_label.to(device)
        batch_size = real_img.size(0)

        # ----------------- Train Discriminator -----------------
        real_targets = torch.ones(batch_size, device=device)
        fake_targets = torch.zeros(batch_size, device=device)

        d_real = D(real_img, real_label)
        loss_d_real = criterion(d_real, real_targets)

        z = torch.randn(batch_size, nz, device=device)
        fake_img = G(z, real_label)
        d_fake = D(fake_img.detach(), real_label)
        loss_d_fake = criterion(d_fake, fake_targets)

        loss_d = loss_d_real + loss_d_fake
        opt_D.zero_grad()
        loss_d.backward()
        opt_D.step()

        # ----------------- Train Generator -----------------
        output = D(fake_img, real_label)
        loss_g = criterion(output, real_targets)

        opt_G.zero_grad()
        loss_g.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss_D: {loss_d.item():.4f}  Loss_G: {loss_g.item():.4f}")

    # ----------------- Save Samples -----------------
    if (epoch + 1) % 10 == 0:
        G.eval()
        z = torch.randn(16, nz, device=device)
        labels = torch.full((16,), minority_class, dtype=torch.long, device=device)
        with torch.no_grad():
            fake_imgs = G(z, labels)
        save_image(fake_imgs * 0.5 + 0.5, f"generated_minority/fake_{epoch+1}.png")
        G.train()


# ------------------------- Final Generation -------------------------
print("\nGenerating 10 final samples...")
G.eval()
os.makedirs("final_generated", exist_ok=True)
with torch.no_grad():
    for i in range(10):
        z = torch.randn(1, nz, device=device)
        label = torch.tensor([minority_class], dtype=torch.long, device=device)
        fake_img = G(z, label)
        save_image(fake_img * 0.5 + 0.5, f"final_generated/minority_{i+1}.png")
print("Done: 10 images saved to final_generated/")