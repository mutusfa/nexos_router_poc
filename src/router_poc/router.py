from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import pickle
import random

import dspy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from router_poc import settings as S
from router_poc.vectors import embed


class BaseStrength(ABC):
    def __init__(self, model_path: Path = None, n_estimators: int = 100):
        self.model_path = model_path
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def _to_path(self, prefix: str, llm_name_a: str, llm_name_b: str | None= None) -> Path:
        if llm_name_b is None:
            return S.MODELS_DIR / prefix / f"{llm_name_a.replace('/', '_')}.pkl"
        return S.MODELS_DIR / prefix / f"{llm_name_a.replace('/', '_')}_{llm_name_b.replace('/', '_')}.pkl"

    def __call__(self, prompt: str):
        return self.predict_proba(embed(prompt).reshape(1, -1))

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path = None):
        path = path or self.model_path
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path = None):
        path = path or self.model_path
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    @abstractmethod
    def train(self, embeddings_path: Path, labels_path: Path):
        pass


class AbsoluteStrength(BaseStrength):
    def __init__(self, llm_name: str, model_path: Path = None, n_estimators: int = 100):
        self.llm_name = llm_name
        model_path = model_path or self._to_path("absolute_strength", llm_name)
        super().__init__(model_path, n_estimators)

    def __call__(self, prompt: str) -> float:
        return super().__call__(prompt)[0][1]

    def train(self, embeddings_path: Path, labels_path: Path):
        embeddings = pd.read_parquet(embeddings_path)
        labels = pd.read_parquet(labels_path)
        labels = labels.query("model == @self.llm_name")
        df = pd.merge(embeddings, labels, on="prompt")
        assert df.shape[0] == labels.shape[0]
        X = np.stack(df["embedding"].values)
        y = df["exact_match"].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        print(f"F1 score: {f1_score(y_test, y_pred)}")
        self.save()


class PairwiseStrength(BaseStrength):
    def __init__(
        self, llm_a: str, llm_b: str, model_path: Path = None, n_estimators: int = 100
    ):
        self.llm_a = llm_a
        self.llm_b = llm_b
        model_path = model_path or self._to_path("pairwise_strength", llm_a, llm_b)
        super().__init__(model_path, n_estimators)

    def __call__(self, prompt: str) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 1.0)  # TODO: Implement proper prediction

    def train(self, embeddings_path: Path, labels_path: Path):
        embeddings = pd.read_parquet(embeddings_path)
        labels = pd.read_parquet(labels_path)
        labels_a = labels.query("model == @self.llm_a")
        labels_b = labels.query("model == @self.llm_b")
        labels = pd.merge(labels_a, labels_b, on="prompt", suffixes=("_a", "_b"))
        # 0 - both wrong, 1 - a right, b wrong, 2 - a wrong, b right, 3 - both right
        labels["target"] = (
            labels["exact_match_a"] + 2 * labels["exact_match_b"]
        ).astype(int)
        labels = pd.merge(embeddings, labels, on="prompt")
        if labels.empty:
            print(f"No labels found for {self.llm_a} and {self.llm_b}")
            return
        X = np.stack(labels["embedding"].values)
        y = labels["target"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        print(f"F1 score: {f1_score(y_test, y_pred, average='macro')}")
        self.save()


class RelativeStrength(BaseStrength):
    def __init__(self, llm_a: str, llm_b: str, model_path: Path = None, n_estimators: int = 100):
        self.llm_a = llm_a
        self.llm_b = llm_b
        model_path = model_path or self._to_path("relative_strength", llm_a, llm_b)
        super().__init__(model_path, n_estimators)

    def train(self, embeddings_path: Path, labels_path: Path):
        embeddings = pd.read_parquet(embeddings_path)
        labels = pd.read_parquet(labels_path)
        labels_a = labels.query("model == @self.llm_a")
        labels_b = labels.query("model == @self.llm_b")
        labels = pd.merge(labels_a, labels_b, on="prompt", suffixes=("_a", "_b"))
        # -1 - a wrong, b right, 0 - tie, 1 - a right, b wrong
        labels["target"] = labels["exact_match_a"] - labels["exact_match_b"]
        labels = pd.merge(embeddings, labels, on="prompt")
        if labels.empty:
            print(f"No labels found for {self.llm_a} and {self.llm_b}")
            return
        X = np.stack(labels["embedding"].values)
        y = labels["target"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        print(f"F1 score: {f1_score(y_test, y_pred, average='macro')}")
        self.save()


class AbsoluteStrengthRouter:
    def __init__(self, llms: list[str]):
        self.llms = llms
        self.models = {llm: AbsoluteStrength(llm) for llm in llms}

    def init(self):
        for model in self.models.values():
            model.load()

    def __call__(self, prompt: str):
        probas = np.array([model(prompt) for model in self.models.values()])
        return self.llms[np.argmax(probas)]


class RandomRouter:
    def __init__(self, llms: list[str]):
        self.llms = llms

    def init(self):
        pass

    def __call__(self, prompt: str):
        return random.choice(self.llms)


class RoutedPrompt(dspy.Signature):
    prompt: str = dspy.InputField()
    llm_name: str = dspy.OutputField()
    response: str = dspy.OutputField()


class Router(dspy.Module):
    def __init__(self, llms: list[str], model_type: str):
        self.llms = {llm: None for llm in llms}
        self.model_type = model_type
        match model_type:
            case "absolute":
                self.router = AbsoluteStrengthRouter(llms)
            case "random":
                self.router = RandomRouter(llms)
            case _:
                raise NotImplementedError(f"Model type {model_type} not implemented")

    def _translate_llm_name(self, llm_name: str) -> str:
        llm_name = llm_name.replace("google", "gemini")
        llm_name = llm_name.replace("mistralai", "mistral")
        return llm_name

    @cached_property
    def init(self):
        self.router.init()
        for llm in self.llms:
            self.llms[llm] = dspy.LM(self._translate_llm_name(llm))

    def forward(self, prompt: str):
        self.init
        llm_name = self.router(prompt)
        return RoutedPrompt(prompt=prompt, llm_name=llm_name, response=self.llms[llm_name](prompt)[0])