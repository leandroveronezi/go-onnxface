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
prebuilt shared libraries per platform. `Recognizer.DownloadModels` fetches the right
one for your OS/arch automatically, the same way go-recognizer's `DownloadModels`
fetches dlib's model files -- no manual ".so path" step.

**Status**: early development.
- ✅ Detection (`yunet.Detector`) -- validated against a real `cv2.FaceDetectorYN`
  run (box/landmarks/score match within ~1px/0.0005).
- ✅ Recognition (`sface.Recognizer`) -- `Align`/`Feature` validated against a
  real `cv2.FaceRecognizerSF` run (same-person cosine ~1.0, different-person
  cosine ~0.11-0.12 on both implementations, well below SFace's ~0.363
  same-person threshold).
- ✅ Easy API (`Recognizer`, file-path driven, auto-downloading) and low-level
  API (`Engine`, `Compare`, working with `image.Image`) -- see Usage below.
- ⏳ Liveness/anti-spoof support -- planned for later.

## Usage

The easy way -- mirrors go-recognizer: point at a directory, everything else
(onnxruntime shared library, detection/recognition models) is fetched into it
automatically and only re-downloaded if missing:

```go
import "github.com/leandroveronezi/go-onnxface"

rec := &onnxface.Recognizer{}

if err := rec.DownloadModels("models"); err != nil {
    // ...
}
if err := rec.Init("models"); err != nil {
    // ...
}
defer rec.Close()

rec.AddImageToDataset("amy.jpg", "Amy")

result, err := rec.Classify("photo.jpg")
if err != nil {
    // no face, or no match within rec.Tolerance
}
fmt.Println(result.Id, result.Distance, result.Confidence)
```

`Recognizer.Tolerance` defaults to 1.128 (OpenCV's suggested SFace L2 threshold)
after `Init`; tune it for your own deployment, same idea as go-face/go-recognizer's
`Tolerance`. `SaveDataset`/`LoadDataset` persist `Dataset` to/from a JSON file.

Lower-level control -- work with `image.Image` directly, pick your own
detector/recognizer, or run detection/recognition as separate steps:

```go
import (
    "github.com/leandroveronezi/go-onnxface"
    "github.com/leandroveronezi/go-onnxface/sface"
    "github.com/leandroveronezi/go-onnxface/yunet"
)

onnxface.InitEnvironment("/path/to/libonnxruntime.so")
defer onnxface.CloseEnvironment()

det, _ := yunet.NewDetector("models/face_detection_yunet_2023mar.onnx")
rec, _ := sface.NewRecognizer("models/face_recognition_sface_2021dec.onnx")

engine := onnxface.NewEngine(det, rec)
defer engine.Close()

results, _ := engine.Recognize(img) // img is a standard image.Image
for _, r := range results {
    fmt.Println(r.Rectangle, r.Landmarks, r.Score, len(r.Feature))
}

result := onnxface.Compare(results[0].Feature, knownFeature, 1.128)
fmt.Println(result.IsMatch, result.Distance, result.Confidence)
```

`yunet`/`sface` implement the shared `onnxface.FaceDetector`/`FaceRecognizer`
contract (in the `face` subpackage) -- a future engine, e.g. an ArcFace-family
recognizer should a commercially-usable license become available, is a new
subpackage implementing that contract, not a change to existing code.

## Requirements

- Go with cgo support.
- The onnxruntime shared library (version 1.26.0) for your platform.
  `Recognizer.DownloadModels` fetches it automatically for linux/amd64,
  linux/arm64, darwin/arm64, windows/amd64 and windows/arm64 (verified
  against the actual [v1.26.0 release assets](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0);
  darwin/amd64 has no prebuilt for this version). For anything else, download
  it yourself and point `InitEnvironment` at it directly.

## Development

Tests that need the onnxruntime shared library read its path from the
`ONNXFACE_ORT_LIB` environment variable and skip themselves if it's unset:

```bash
curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz
tar xzf ort.tgz
ONNXFACE_ORT_LIB="$PWD/onnxruntime-linux-x64-1.26.0/lib/libonnxruntime.so.1.26.0" go test ./...
```

`TestDownloadModels` additionally does a real network download to verify the
auto-download path itself works end to end.

## License

[MIT](LICENSE)
