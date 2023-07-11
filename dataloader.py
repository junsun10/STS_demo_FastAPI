import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch
from itertools import chain

import pytorch_lightning as pl

from dataset import Dataset


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, sentence1, sentence2):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

        self.sentence1 = sentence1
        self.sentence2 = sentence2

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = "[SEP]".join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
            data.append(outputs["input_ids"])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            self.train_inputs, self.train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # 검증데이터 세팅
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            # test_data = pd.read_csv(self.test_path)
            # test_inputs, test_targets = self.preprocessing(test_data)
            # self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.DataFrame({"id": "", "sentence_1": [self.sentence1], "sentence_2": [self.sentence2]})
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        """_summary_

        토크나이징 된 데이터를 샘플링

        Returns:
            _type_: _description_
        """
        origin_data = pd.DataFrame({"data": self.train_inputs, "label": list(chain.from_iterable(self.train_targets))})
        # 소수점 첫째자리 짝수면 600개, 홀수면 60개 샘플링
        train_data = pd.concat(
            [origin_data[origin_data.label == i / 10].sample(600, replace=True) for i in range(0, 51, 2)]
            + [origin_data[origin_data.label == i / 10].sample(60, replace=True) for i in range(5, 46, 10)]
        )
        self.train_dataset = Dataset(train_data.data.tolist(), [[i] for i in train_data.label])
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
