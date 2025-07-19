import asyncio
import os
import weakref
from typing import Any, Literal

import aiohttp
import msgpack
import numpy as np
from livekit.agents import (APIConnectionError, APIConnectOptions,
                            APIStatusError, APITimeoutError, tokenize, tts,
                            utils)
from livekit.agents.types import (DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN,
                                  NotGivenOr)
from livekit.agents.utils import is_given

from .log import logger
from .types import TTSClientEosMessage, TTSClientTextMessage, TtsStreamingQuery
from .utils import float32_to_int16_pcm


class TTS(tts.TTS):
    """Client for Kyutai Moshi rust server

    Notes
    -------------
    * Uses msgpack and websockets for transport

    Configuration
    -------------
    * Host URL: ``base_url`` or environ ``KYUTAI_BASE_URL``.  
    * Auth token: ``auth_id`` or ``KYUTAI_API_KEY`` (defaults to “public_token”).  
    * Voices: plain strings from server voicepack or use server defined fallback.  
    * Quality: temperature, top_k, cfg_alpha, seed, max_seq_len.  
    * Output:  ``format="OggOpus"`` (default, 48 k) or ``"Pcm"`` (raw 16-bit PCM).
    """

    TEXT_TO_SPEECH_PATH = "/api/tts-streaming"
    SAMPLE_RATE = 24_000
    NUM_CHANNELS = 1
    API_AUTH_HEADER = "kyutai-api-key"
    DEFAULT_API_TOKEN = "public_token"

    def __init__(
        self,
        base_url: str | None = None,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        path: str = TEXT_TO_SPEECH_PATH,
        sample_rate: int = SAMPLE_RATE,
        num_channels: int = NUM_CHANNELS,
        seed: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        voice: str | None = None,
        # FIXME: Unknown `voices`
        voices: list[str] | None = None,
        max_seq_len: int | None = None,
        cfg_alpha: float | None = None,
        # Authorization field
        auth_id: str = DEFAULT_API_TOKEN,
        # These formats will send raw audio bytes only, instead of using `msgpack`
        #   to include extra information about the text placement
        format: Literal[
            "OggOpus",
            "Pcm"
        ] = "OggOpus",
        http_session: aiohttp.ClientSession | None = None,
    ):
        sample_rate = sample_rate if format != "OggOpus" else 48000

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels
        )

        self.path = path
        self.base_url = base_url or os.environ.get("KYUTAI_BASE_URL")
        assert self.base_url is not None, "`url` is needed (env: KYUTAI_BASE_URL)"

        auth_id = auth_id or os.environ.get(
            "KYUTAI_API_KEY",
            self.DEFAULT_API_TOKEN
        )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(
                ignore_punctuation=True)

        self.word_tokenizer = word_tokenizer

        self._opts = TtsStreamingQuery(
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            voice=voice,
            voices=voices,
            max_seq_len=max_seq_len,
            cfg_alpha=cfg_alpha,
            auth_id=auth_id,
            format=format,
        )

        self._session: aiohttp.ClientSession | None = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()

        assert self.base_url is not None
        logger.debug('connecting to %s', self.base_url)

        return await asyncio.wait_for(
            session.ws_connect(
                self._opts.to_url(self.base_url, self.path),
                headers={
                    self.API_AUTH_HEADER: f"{self._opts.auth_id}"
                },
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )

        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (str): TTS voice to use.
        """
        if is_given(voice):
            self._opts.voice = voice

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._pools = self._tts._pool
        self._opts = self._tts._opts

        self.is_pcm = (self._opts.format == "Pcm")

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        mime_type = "audio/pcm" if self.is_pcm else "audio/opus"

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=mime_type,
        )

        closing_ws: bool = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            speak_msg = TTSClientTextMessage(
                type="Text", text=f"{self.input_text}"
            )
            data = msgpack.packb(speak_msg.model_dump())

            logger.debug("sending whole text: '''\n%s\n'''", self.input_text)

            await ws.send_bytes(data)

            # Always flush after a segment
            flush_msg = TTSClientEosMessage()
            data: Any = msgpack.packb(flush_msg.model_dump())
            closing_ws = True
            await ws.send_bytes(data)

            output_emitter.flush()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        break

                    raise APIStatusError(
                        f"Kyutai websocket connection closed unexpectedly ({msg.type})")

                if msg.type is aiohttp.WSMsgType.BINARY:
                    data = msg.data

                    # XXX: Server sends f32
                    if self.is_pcm:
                        data = float32_to_int16_pcm(
                            np.frombuffer(
                                data,
                                dtype=np.float32
                            )
                        ).tobytes()

                    output_emitter.push(data)
            # done:
            logger.debug("finished audio recv_task")
            output_emitter.flush()

        async with self._pools.connection(timeout=self._conn_options.timeout) as ws:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_task(ws))
                tg.create_task(recv_task(ws))


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._pools = self._tts._pool
        self._opts = self._tts._opts
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

        self.is_pcm = (self._opts.format == "Pcm")

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        mime_type = "audio/pcm" if self.is_pcm else "audio/opus"

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=mime_type,
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into WordStreams and sends them into _segments_ch
            word_stream = None
            async for word in self._input_ch:
                if isinstance(word, str):
                    if word_stream is None:
                        word_stream = self._tts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)
                    word_stream.push_text(word)
                elif isinstance(word, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            self._segments_ch.close()

        async def _run_segments() -> None:
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, output_emitter)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_tokenize_input())
                tg.create_task(_run_segments())
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(
        self, word_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        closing_ws: bool = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            async for word in word_stream:
                if not word.token:
                    continue  # Don't send empty messages

                speak_msg = TTSClientTextMessage(
                    type="Text", text=f"{word.token} "
                )
                data: Any = msgpack.packb(speak_msg.model_dump())

                logger.debug("sending word: '%s'", word.token)

                await ws.send_bytes(data)

            # Always flush after a segment
            flush_msg = TTSClientEosMessage()
            data: Any = msgpack.packb(flush_msg.model_dump())
            closing_ws = True
            await ws.send_bytes(data)
            output_emitter.flush()

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        break

                    raise APIStatusError(
                        f"Kyutai websocket connection closed unexpectedly ({msg.type})")

                if msg.type is aiohttp.WSMsgType.BINARY:
                    data = msg.data

                    # XXX: Server sends f32
                    if self.is_pcm:
                        data = float32_to_int16_pcm(
                            np.frombuffer(
                                data,
                                dtype=np.float32
                            )
                        ).tobytes()

                    output_emitter.push(data)
            # done:
            logger.debug("finished audio recv_task")
            output_emitter.flush()

        async with self._pools.connection(timeout=self._conn_options.timeout) as ws:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_task(ws))
                tg.create_task(recv_task(ws))
