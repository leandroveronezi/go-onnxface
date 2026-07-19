## go-onnxface

[![CI](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml/badge.svg)](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/leandroveronezi/go-onnxface.svg)](https://pkg.go.dev/github.com/leandroveronezi/go-onnxface)
![MIT Licensed](https://img.shields.io/github/license/leandroveronezi/go-onnxface.svg)

Face detection and recognition for Go, backed by [ONNX Runtime](https://onnxruntime.ai)
instead of dlib. Sibling project to [go-face](https://github.com/leandroveronezi/go-face)/
[go-recognizer](https://github.com/leandroveronezi/go-recognizer), aimed at better accuracy
under real-world conditions (pose, lighting) than dlib's 2017-era model, using multiple
interchangeable engines -- each a subpackage implementing a shared contract (`face.FaceDetector`
or `face.FaceRecognizer`), so adding one doesn't change how the others are used.

Unlike dlib, ONNX Runtime doesn't need to be compiled from source: Microsoft publishes
prebuilt shared libraries per platform. `Recognizer.DownloadModels` fetches the right
one for your OS/arch automatically, the same way go-recognizer's `DownloadModels`
fetches dlib's model files -- no manual ".so path" step.

### Engines

Every model here was picked for having an explicit, commercially-usable license on the
*published weights* -- not just permissive code wrapping weights trained on a
research-only dataset (MS1M/CASIA-WebFace/VGGFace2 and similar), which is the trap
most "MIT-licensed" face recognition repos fall into. See each row's link for how it
was verified.

| Kind | Package | Model | License | Notes |
|------|---------|-------|---------|-------|
| Detection | `yunet` | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT | Default. Fixed 640x640 input (letterboxed). |
| Detection | `centerface` | [CenterFace](https://github.com/Star-Clouds/CenterFace) | MIT | Dynamic input size (resized to a multiple of 32, no letterbox distortion). |
| Recognition | `sface` | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache-2.0 | Only recognition model found so far with an explicit commercial grant on the weights -- see Licensing below. |
| Recognition | `arcface` | any ArcFace-family ONNX export | *depends on your weights* | A bridge, not a model: ships no weights, downloads none. See Licensing below before using it. |

**Status**: early development.
- ✅ Detection (`yunet.Detector`, `centerface.Detector`) -- each validated against
  its own real Python/OpenCV reference run (box/landmarks/score match within ~1-2px).
- ✅ Recognition (`sface.Recognizer`) -- `Align`/`Feature` validated against a
  real `cv2.FaceRecognizerSF` run (same-person cosine ~1.0, different-person
  cosine ~0.11-0.12 on both implementations, well below SFace's ~0.363
  same-person threshold).
- ✅ Easy API (`Recognizer`, file-path driven, auto-downloading) and low-level
  API (`Engine`, `Compare`, working with `image.Image`) -- see Usage below.
- ✅ `arcface` bridge (code only, bring your own weights) -- validated locally
  against a real InsightFace `buffalo_l` (`w600k_r50.onnx`) run (same-person
  cosine ~1.0, different-person cosine ~-0.03); not part of CI since it needs
  weights this repo can't ship.
- ⏳ Liveness/anti-spoof support -- planned for later.

### Licensing: why so few recognition models

Face *detection* models (YuNet, CenterFace, RetinaFace, MTCNN, ...) train on WIDER
FACE -- bounding boxes only, no identity labels -- so licensing them commercially is
usually uneventful. Face *recognition* models need identity-labeled data, and nearly
every well-known one (ArcFace, GhostFaceNets, FaceNet, VGG-Face, Buffalo_L, ...) is
trained on MS1M/CASIA-WebFace/VGGFace2-lineage datasets released "for research
purposes only" -- that restriction carries over to the weights regardless of the
*code's* license (MIT code around non-commercial weights is still non-commercial to
actually use). SFace is the one exception found so far: OpenCV Zoo distributes its
specific weights under an explicit Apache-2.0 grant.

The `arcface` package exists for the rest of that family anyway, but deliberately
as *just code*: it doesn't ship or download any `.onnx` file, because doing so would
mean this MIT-licensed repository redistributing someone else's non-commercial-only
weights -- the wrapper's own license doesn't change what license the weights carry.
Bring your own file, and make sure you have the rights to it for how you intend to
use it: research use of InsightFace's published `buffalo_l`/`antelopev2` is generally
fine under their terms, commercial use needs their paid license (see their commercial
licensing page). This isn't legal advice, just how the ecosystem's licensing actually
works in practice -- when unsure, ask a lawyer, not this README.

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

`Recognizer` always uses `yunet`+`sface` internally (`DownloadModels`/`Init`'s
defaults). `Recognizer.Tolerance` defaults to 1.128 (OpenCV's suggested SFace L2
threshold) after `Init`; tune it for your own deployment, same idea as
go-face/go-recognizer's `Tolerance`. `SaveDataset`/`LoadDataset` persist `Dataset`
to/from a JSON file.

Lower-level control -- work with `image.Image` directly, pick your own
detector/recognizer (e.g. `centerface` instead of `yunet`), or run detection/
recognition as separate steps:

```go
import (
    "github.com/leandroveronezi/go-onnxface"
    "github.com/leandroveronezi/go-onnxface/centerface"
    "github.com/leandroveronezi/go-onnxface/sface"
)

onnxface.InitEnvironment("/path/to/libonnxruntime.so")
defer onnxface.CloseEnvironment()

det, _ := centerface.NewDetector("models/centerface.onnx")
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

`yunet`/`centerface`/`sface`/`arcface` implement the shared `onnxface.FaceDetector`/
`FaceRecognizer` contract (in the `face` subpackage) -- a future engine is a new
subpackage implementing that contract, not a change to existing code. See
`examples/` for complete, runnable programs.

Using `arcface` (see its Licensing note above) means supplying your own model file
plus its tensor names, since those vary across ArcFace exports:

```go
rec, _ := arcface.NewRecognizer("/path/to/your/model.onnx", arcface.Config{
    InputName:  "input.1", // inspect your own file -- see the arcface package doc
    OutputName: "683",
})
```

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
auto-download path itself works end to end. `arcface`'s test needs its own
locally-provided (correctly licensed) model and skips itself without it -- see
the doc comment on `TestRecognizerAgainstLocalModel` in `arcface/recognizer_test.go`.

## License

[MIT](LICENSE)
