import urllib.parse
from typing import (Annotated, Literal, Union)

from pydantic import BaseModel, Field, TypeAdapter


class TtsStreamingQuery(BaseModel):
    # See moshi-rs/moshi-server/
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    format: str = "OggOpus"
    voice: str | None = None
    voices: list[str] | None = None
    max_seq_len: int | None = None
    cfg_alpha: float | None = None
    auth_id: str | None = None

    @staticmethod
    def url_escape(value: object) -> str:
        return urllib.parse.quote(str(value), safe="")

    def to_url_params(self) -> str:
        params = self.model_dump()
        return "?" + "&".join(
            f"{key}={TtsStreamingQuery.url_escape(value)}"
            for key, value in params.items()
            if value is not None
        )

    def to_url(self, base_url: str, path: str):
        if base_url.startswith("http"):
            base_url = base_url.replace("http", "ws", 1)
        if base_url.startswith("https"):
            base_url = base_url.replace("https", "wss", 1)

        if path.startswith("/"):
            path = path.replace("/", "", 1)

        return f"{base_url}/{path}{self.to_url_params()}"


class TTSClientTextMessage(BaseModel):
    """Message sent to the TTS server saying we to turn this text into speech."""
    type: Literal["Text"] = "Text"
    text: str


class TTSClientVoiceMessage(BaseModel):
    type: Literal["Voice"] = "Voice"
    embeddings: list[float]
    shape: list[int]


class TTSClientEosMessage(BaseModel):
    """Message sent to the TTS server saying we are done sending text."""
    type: Literal["Eos"] = "Eos"


class TTSTextMessage(BaseModel):
    type: Literal["Text"]
    text: str
    start_s: float
    stop_s: float


class TTSAudioMessage(BaseModel):
    type: Literal["Audio"]
    pcm: list[float]


class TTSErrorMessage(BaseModel):
    type: Literal["Error"]
    message: str


class TTSReadyMessage(BaseModel):
    type: Literal["Ready"]


TTSClientMessage = Annotated[
    Union[TTSClientTextMessage, TTSClientVoiceMessage, TTSClientEosMessage],
    Field(discriminator="type"),
]
TTSClientMessageAdapter = TypeAdapter(TTSClientMessage)

TTSMessage = Annotated[
    Union[TTSTextMessage, TTSAudioMessage, TTSErrorMessage, TTSReadyMessage],
    Field(discriminator="type"),
]
TTSMessageAdapter = TypeAdapter(TTSMessage)
