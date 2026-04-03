import torch
import torch.nn as nn
import torch.nn.functional as F

#define the u-net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequence(x)


# Attention Gate: learns to suppress irrelevant background regions
# and focus on small, meaningful landslide areas in the skip connections.
# g = gating signal (from decoder/bottleneck), x = skip connection (from encoder)
class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        # 1x1 convs to project both signals to the same intermediate space
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        # Produces a single-channel attention map (alpha) between 0 and 1
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g typically has a smaller spatial size; upsample to match x
        g_proj = self.W_g(F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True))
        x_proj = self.W_x(x)
        # Add projections, apply relu, then compute the attention map
        alpha = self.psi(self.relu(g_proj + x_proj))
        # Multiply the attention map back onto the skip connection
        return x * alpha


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=14, out_channels=1):
        super().__init__()

        # --- Encoder (Contracting Path) ---
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.pool  = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(256, 512)

        # --- Attention Gates (one per skip connection) ---
        # g_channels = channels from decoder/bottleneck side
        # x_channels = channels from the corresponding encoder skip connection
        self.att3 = AttentionGate(g_channels=512, x_channels=256, inter_channels=128)
        self.att2 = AttentionGate(g_channels=256, x_channels=128, inter_channels=64)
        self.att1 = AttentionGate(g_channels=128, x_channels=64,  inter_channels=32)

        # --- Decoder (Expanding Path) ---
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64,  64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))

        # Bottleneck
        x  = self.bottleneck(self.pool(c3))

        # Decoder with attention-gated skip connections
        x  = self.upsample(x)
        c3 = self.att3(g=x, x=c3)
        x  = self.up3(torch.cat([x, c3], dim=1))

        x  = self.upsample(x)
        c2 = self.att2(g=x, x=c2)
        x  = self.up2(torch.cat([x, c2], dim=1))

        x  = self.upsample(x)
        c1 = self.att1(g=x, x=c1)
        x  = self.up1(torch.cat([x, c1], dim=1))

        return torch.sigmoid(self.final_conv(x))


# Keep original UNet as a fallback alias
UNet = AttentionUNet
