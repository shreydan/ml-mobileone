import torch
import torch.nn as nn
from mobileone import mobileone, reparameterize_model
from torch.utils.mobile_optimizer import optimize_for_mobile


class Config:
    mobileone_variant = "s1"
    load_ckpt = False


class Model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.backbone = mobileone(variant=Config.mobileone_variant)
        if Config.load_ckpt:
            checkpoint = torch.load(
                f"./checkpoints/mobileone_{Config.mobileone_variant}_unfused.pth.tar"
            )
            self.backbone.load_state_dict(checkpoint)

        in_ft = self.backbone.linear.in_features
        self.backbone.linear = nn.Identity()

        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_ft, 10))

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = Model()
    model.eval()
    eval_model = reparameterize_model(model)
    # not working coz of reparameterization
    # script_model = torch.jit.trace(eval_model)
    # using torch.jit.trace instead
    # only downside is gotta preset the input size
    trace_model = torch.jit.trace(eval_model, torch.rand(1, 3, 128, 128))

    optimized_model = optimize_for_mobile(trace_model)
    optimized_model._save_for_lite_interpreter("model.ptl")
