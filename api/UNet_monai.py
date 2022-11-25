import cv2
import torch
import gc
import numpy as np
from args import *
from loss import *
from metrics import *
import pytorch_lightning as pl
from monai.networks.nets import DynUNet
from pathlib import Path
import matplotlib.pyplot as plt


class Unet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_dice, self.best_dice_mean = (0,) * 2
        self.build_model()
        self.loss = LossSpaceNet7()
        self.dice = DiceSpaceNet7(n_class=self.args.out_channels)

    def forward(self, img):
        return torch.argmax(self.model(img, dim=1))

    def training_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        return loss

    def validation_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        self.dice.update(logits, lbl, loss)

    def predict_step(self, batch, batch_idx):
        img, lbl = batch
        preds = self.model(img)
        preds = (nn.Sigmoid()(preds) > 0.5).int()
        imgs_np = img.detach().cpu().numpy()
        lbls_np = lbl.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        storage_dir = "/storage"
        print(f"'{storage_dir}' exists? {Path('{storage_dir}').exists()}")
        print(f"'{storage_dir}' is dir? {Path('{storage_dir}').is_dir()}")
        for idx, (img, gt, pred) in enumerate(zip(imgs_np, lbls_np, preds_np)):
            id = f"{idx:05d}"

            fig = plt.figure(figsize=(20, 10), dpi=300)

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title("Ground Truth")
            ax1.imshow(lbls_np[idx][0], cmap="gray")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title("Prediction")
            ax2.imshow(preds_np[idx][0], cmap="gray")

            # Save the full figure...
            fig.savefig(f"/storage/{id}.png")

    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        gc.collect()

    def validation_epoch_end(self, outputs):
        dice, loss = self.dice.compute()
        dice_mean = dice.item()
        self.dice.reset()

        if dice_mean >= self.best_dice_mean:
            self.best_mean_dice = dice_mean

        metrics = {}
        metrics.update({"Mean_Dice": round(dice_mean, 2)})
        metrics.update({"Highest": round(self.best_dice_mean, 2)})
        metrics.update({"val_loss": round(loss.item(), 4)})

        print(
            f"Val_Performace: Mean_Dice {metrics['Mean_Dice']}, Val_Loss {metrics['val_loss']}"
        )
        self.log("dice_mean", dice_mean)
        torch.cuda.empty_cache()
        gc.collect()

    def build_model(self):
        # uncomment this line if you want to try the UNet implemented from scratch
        # self.model = NNUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
        self.model = DynUNet(
            spatial_dims=2,
            in_channels=self.args.in_channels,
            out_channels=self.args.out_channels,
            kernel_size=self.args.kernels,
            strides=self.args.strides,
            upsample_kernel_size=self.args.strides[1:],
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weigh_decay,
        )
