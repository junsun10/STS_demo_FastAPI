import torch
import pytorch_lightning as pl

from dataloader import Dataloader


def load_model():
    return torch.load("snunlp09316.pt")
    # return torch.load("klue09147.pt")


def predict(model_name, input1, input2):
    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator="gpu")

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = load_model()
    dataloader = Dataloader(model_name, 1, True, "",
                            "", "", "", input1, input2)

    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    answer = round(predictions[0].item(), 1)
    if answer < 0:
        answer = 0.0
    elif answer > 5:
        answer = 5.0

    return str(answer)
