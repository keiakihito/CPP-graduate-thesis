# FFmpeg Binary Layer (Serverless Lambda)

This documents why and how we ship FFmpeg as a Lambda layer, and how the code uses it.

## Why a layer?
- Lambda runtimes do not include FFmpeg.
- Packaging FFmpeg inside each function zip increases size and slows deploys/cold starts.
- A Lambda layer lets us share the binary across functions and keep the function bundle small.

## What we ship
- Directory: `backend/layers/ffmpeg/bin/ffmpeg`
- Serverless layer config: `serverless.yml` adds a `ffmpeg` layer pointing to `layers/ffmpeg`.
- The layer is named `${service}-${stage}-ffmpeg` and is compatible with `nodejs20.x`.
- Serverless exposes the layer to functions via the logical ID `FfmpegLambdaLayer`.

## How functions use it
- `serverless.yml` attaches the layer to the `http` function:
  - `layers: - { Ref: FfmpegLambdaLayer }`
- `FFMPEG_PATH` environment variable is set to `/opt/bin/ffmpeg`.
- Code reads `process.env.FFMPEG_PATH ?? "ffmpeg"` to resolve the binary:
  - `src/config/env.ts` sets `ffmpegPath`.
  - Health check `checkFfmpegAvailable` and FFmpeg runner classes use that path.
- On Lambda, the layer places the binary at `/opt/bin/ffmpeg`. Locally, you can rely on a system FFmpeg if `FFMPEG_PATH` is not set.

## Deploy flow
1) Ensure `backend/layers/ffmpeg/bin/ffmpeg` is present and executable.
2) Deploy via Serverless (manually or CI): the layer is packaged and published, and the function references `FfmpegLambdaLayer`.
3) The function runs with `FFMPEG_PATH=/opt/bin/ffmpeg` and uses the layer binary.

## Local development
- If you do not have FFmpeg in PATH, set `FFMPEG_PATH` in your `.env` to your local binary path.
- Otherwise the fallback `ffmpeg` command is used.

## Troubleshooting
- Validation errors about layer reference: ensure the function `layers` entry uses `{ Ref: FfmpegLambdaLayer }`.
- Permission or missing binary errors: confirm the binary is executable and the layer directory is included in the deploy.
- If you change the layer path or name, update both `serverless.yml` (layer path/name) and the function `layers` reference.
