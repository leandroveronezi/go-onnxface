## go-onnxface

[![CI](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml/badge.svg)](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/leandroveronezi/go-onnxface.svg)](https://pkg.go.dev/github.com/leandroveronezi/go-onnxface)
![MIT Licensed](https://img.shields.io/github/license/leandroveronezi/go-onnxface.svg)

Face detection and recognition for Go, backed by [ONNX Runtime](https://onnxruntime.ai)
instead of dlib. Sibling project to [go-face](https://github.com/leandroveronezi/go-face)/
[go-recognizer](https://github.com/leandroveronezi/go-recognizer), aimed at better accuracy
under real-world conditions (pose, lighting) using modern, commercially-licensed models:

- **Detection**: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT)
- **Recognition**: [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache-2.0)

Unlike dlib, ONNX Runtime doesn't need to be compiled from source: Microsoft publishes
prebuilt shared libraries per platform, so setup is a download instead of a build.

**Status**: early development. Detection and recognition are being built out; a
higher-level API and liveness/anti-spoof support are planned for later.

## Requirements

- Go with cgo support.
- The onnxruntime shared library (version 1.26.0) for your platform, from the
  [official releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0).
  This is a prebuilt binary -- no compilation needed. Point `Init` at it:

```go
import "github.com/leandroveronezi/go-onnxface"

err := onnxface.Init("/path/to/libonnxruntime.so")
if err != nil {
    // ...
}
defer onnxface.Close()
```

## Development

Tests that need the onnxruntime shared library read its path from the
`ONNXFACE_ORT_LIB` environment variable and skip themselves if it's unset:

```bash
curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz
tar xzf ort.tgz
ONNXFACE_ORT_LIB="$PWD/onnxruntime-linux-x64-1.26.0/lib/libonnxruntime.so.1.26.0" go test ./...
```

## License

[MIT](LICENSE)
