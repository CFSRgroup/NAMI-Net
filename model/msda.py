import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from timm.layers import DropPath

# References & Acknowledgments
# Portions of the code in this file were adapted from the following source:
# https://github.com/fundamentalvision/Deformable-DETR
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(H_, dtype=torch.float32, device=device) + 0.5,
            torch.arange(W_, dtype=torch.float32, device=device) + 0.5,
            indexing="ij"
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    return reference_points[:, :, None]


class CTIModule(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, drop_path=0.1):
        super(CTIModule, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = MultiScaleDeformableAttention(
            embed_dims=embed_dim,
            num_heads=num_heads,
            num_levels=3,
            num_points=4,
            batch_first=True,
            im2col_step=256*2
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, f3, f4, f5):
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, height * width, channels)

        f3 = f3.reshape(batch_size, -1, channels)
        f4 = f4.reshape(batch_size, -1, channels)
        f5 = f5.reshape(batch_size, -1, channels)

        multi_scale_features = torch.cat([f3, f4, f5], dim=1)

        spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8]], dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        reference_points = get_reference_points(spatial_shapes, x.device)
        reference_points = reference_points.repeat(batch_size, 1, 1, 1)

        query_num = f4.size(1)
        ref_num = reference_points.size(1)

        if ref_num != query_num:
            reference_points = reference_points.permute(0, 2, 3, 1).reshape(-1, 2, ref_num)
            reference_points = nn.functional.interpolate(
                reference_points, size=(query_num,), mode="linear", align_corners=False
            )
            reference_points = reference_points.reshape(batch_size, -1, 2, query_num).permute(0, 3, 1, 2)

        f4 = f4 + x

        attn_output = self.attention(
            query=self.norm(f4),
            key=self.norm(multi_scale_features),
            value=self.norm(multi_scale_features),
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        o = f4 + self.drop_path(self.ffn(self.norm(attn_output)))

        o = o.view(batch_size, channels, height, width)
        o3 = nn.functional.interpolate(o, size=(32, 32), mode='bilinear', align_corners=False) + f3.view(batch_size,channels, 32,32)
        o4 = o
        o5 = nn.functional.interpolate(o, size=(8, 8), mode='bilinear', align_corners=False) + f5.view(batch_size,channels, 8, 8)

        o3 = nn.functional.interpolate(o3, size=(16, 16), mode='bilinear', align_corners=True)
        o4 = o4
        o5 = nn.functional.interpolate(o5, size=(16, 16), mode='bilinear', align_corners=True)

        return o3, o4, o5



