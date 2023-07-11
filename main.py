from typing import Union

from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

import uvicorn

from predict import *

app = FastAPI()
templates = Jinja2Templates(directory="./src")


@app.get("/")
def get_sentense_form(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.post("/predict")
def return_predict(request: Request, sentence1: str = Form(...), sentence2: str = Form(...)):
    similarity = predict("snunlp/KR-ELECTRA-discriminator",
                         sentence1, sentence2)

    return templates.TemplateResponse("predict.html", context={"request": request, "similarity": similarity})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
