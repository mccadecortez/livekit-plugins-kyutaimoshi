# Kyutai-labs's Moshi - LiveKit Agents

Support for near-realtime TTS and STT.

See [https://github.com/kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) and [https://github.com/kyutai-labs/moshi/tree/main/rust](https://github.com/kyutai-labs/moshi/tree/main/rust) for more information.

This repo attempts to interact with moshi's rust server directly, instead of using creating a OpenAI wrapper over the model. See `docker` directory for building the server image.

This code was initalized with the example plugin and used structure of other existing plugins.

## Notes
- Server only supports pcm/opus natively
- Server uses `msgpack` to send info instead of json or protobuf

## Installation

```bash
cd docker
docker build -f Dockerfile --tag=moshi-server
docker run --gpus all --name moshi-server -p $PORT:8080 moshi-server:latest
podman run --replace --device nvidia.com/gpu=all --name moshi-server -p $PORT:8080 moshi-server:latest
cd -
```

```bash
pip install livekit-kyutai-moshi
```

## TODO (PR welcomed):
- STT impl
- Testing