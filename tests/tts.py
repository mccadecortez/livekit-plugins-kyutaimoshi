#!/usr/bin/env python3
"""
test_kyutai_tts.py - minimal reproducible test
"""
import asyncio
import sys
import tempfile
import logging
from pathlib import Path

import aiohttp
from livekit import rtc
from livekit.plugins.kyutaimoshi.tts import TTS


BASE_URL = "ws://mac-ng.local:3000"  # edit to your endpoint
VOICE = "vctk/p238_023.wav"  # sample voice preset
TEXT = """
Good evening my fellow citizens:

This afternoon, following a series of threats and defiant statements, the presence of Alabama National Guardsmen was required on the University of Alabama to carry out the final and unequivocal order of the United States District Court of the Northern District of Alabama. That order called for the admission of two clearly qualified young Alabama residents who happened to have been born Negro.

That they were admitted peacefully on the campus is due in good measure to the conduct of the students of the University of Alabama, who met their responsibilities in a constructive way.

I hope that every American, regardless of where he lives, will stop and examine his conscience about this and other related incidents. This Nation was founded by men of many nations and backgrounds. It was founded on the principle that all men are created equal, and that the rights of every man are diminished when the rights of one man are threatened.

Today we are committed to a worldwide struggle to promote and protect the rights of all who wish to be free. And when Americans are sent to Viet-Nam or West Berlin, we do not ask for whites only. It ought to be possible, therefore, for American students of any color to attend any public institution they select without having to be backed up by troops.

It ought to be possible for American consumers of any color to receive equal service in places of public accommodation, such as hotels and restaurants and theaters and retail stores, without being forced to resort to demonstrations in the street, and it ought to be possible for American citizens of any color to register to vote in a free election without interference or fear of reprisal
"""


async def _run_once(client: TTS, *, stream: bool) -> rtc.AudioFrame:
    """Run a single test and return the decoded PCM frame."""
    if stream:
        async with client.stream() as ts:
            ts.push_text(TEXT)
            ts.end_input()
            frames = [ev.frame async for ev in ts]
    else:
        frames = [ev.frame async for ev in client.synthesize(TEXT)]

    frame = rtc.combine_audio_frames(frames)
    assert len(frame.data) > 0, "empty audio returned"
    assert frame.sample_rate == client.sample_rate
    return frame


async def main() -> None:
    log = logging.getLogger('livekit.plugins.kyutaimoshi')
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))

    async with aiohttp.ClientSession() as http:
        # --- chunked (non-streaming) path
        chunked = await _run_once(
            TTS(base_url=BASE_URL, http_session=http, format="OggOpus"), stream=False
        )

        # --- full-streaming path
        streamed = await _run_once(
            TTS(base_url=BASE_URL, http_session=http,
                voice=VOICE, format="OggOpus"),
            stream=True,
        )

        # save artefacts for external inspection
        with tempfile.TemporaryDirectory(delete=False) as td:
            Path(td, "chunked.wav").write_bytes(chunked.to_wav_bytes())
            Path(td, "streamed.wav").write_bytes(streamed.to_wav_bytes())
            print(
                "✅  Test artefacts written to %s (exact size %d bytes)",
                td,
                len(streamed.data),
            )


if __name__ == "__main__":
    asyncio.run(main())
