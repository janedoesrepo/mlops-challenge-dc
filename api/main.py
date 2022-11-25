import uvicorn
from args import get_main_args
from dataset import SpaceNet7DataModule
from fastapi import FastAPI
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from UNet_monai import Unet

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome fellow ML enthusiast"}


@app.post("/{selected_checkpoint}")
def run_inference(selected_checkpoint: str):

    args = get_main_args()

    callbacks = []
    model = Unet(args)
    model_ckpt = ModelCheckpoint(
        dirpath="./",
        filename="last",
        monitor="dice_mean",
        mode="max",
        save_last=True,
    )
    callbacks.append(model_ckpt)

    dm = SpaceNet7DataModule(args)
    trainer = Trainer(
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=args.num_epochs,
        enable_progress_bar=True,
        gpus=1,
        accelerator="cpu",
        amp_backend="apex",
        profiler="simple",
    )

    trainer.predict(
        model, datamodule=dm, ckpt_path=f"trained_models/{selected_checkpoint}"
    )

    return {"result": "success"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
