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
| Detection | `retinaface` | [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) (resnet50) | MIT | Heaviest of the three (~52MB float16 vs CenterFace's ~7.5MB/YuNet's ~230KB) and, per each project's own self-reported WIDER FACE hard-set numbers (not a unified benchmark, so treat as directional): the most accurate -- RetinaFace resnet50 ~84-90% vs CenterFace ~87% vs YuNet ~75%. Fixed 640x640 input (letterboxed). |
| Recognition | `sface` | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache-2.0 | Only recognition model found so far with an explicit commercial grant on the weights -- see Licensing below. |
| Recognition | `arcface` | any ArcFace-family ONNX export | *depends on your weights* | A bridge, not a model: ships no weights, downloads none. See Licensing below before using it. |

**Status**: early development.
- ✅ Detection (`yunet.Detector`, `centerface.Detector`, `retinaface.Detector`) --
  each validated against its own real Python reference run (box/landmarks/score
  match within ~1-2px). CenterFace's and RetinaFace's `.onnx` files aren't
  straight upstream downloads (see Development below for why and how they were
  produced) and, unlike YuNet/SFace, aren't hosted by their own upstream project
  either -- `centerface.DownloadModel`/`retinaface.DownloadModel` fetch go-onnxface's
  own copies instead (see the Development section).
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

// Each engine fetches only its own model -- using centerface here never
// downloads yunet/sface too, and vice versa.
centerface.DownloadModel("models")

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

No model weights are checked into this repository (see Licensing above for
why, for `arcface` at least -- for the others it's to keep clones small and
avoid GitHub's file-size limits, the same reason DeepFace's own weights live
in a separate release rather than its main repo). Fetch what the tests need
into `models/` before running them:

```bash
curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz
tar xzf ort.tgz
export ONNXFACE_ORT_LIB="$PWD/onnxruntime-linux-x64-1.26.0/lib/libonnxruntime.so.1.26.0"

mkdir -p models
curl -sL -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
curl -sL -o models/centerface.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/centerface.onnx
curl -sL -o models/retinaface.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/retinaface.onnx

go test ./...
```

Tests that need the onnxruntime shared library or a model file skip
themselves if it/they aren't present, so a partial fetch just skips fewer
tests rather than failing the ones that don't need what's missing.
`TestDownloadModels`/`centerface.TestDownloadModel`/`retinaface.TestDownloadModel`
additionally do a real network download into a fresh directory each, to
verify the download path itself (not just a pre-fetched file) works end to
end. `arcface`'s test needs its own locally-provided (correctly licensed)
model and skips itself without it -- see the doc comment on
`TestRecognizerAgainstLocalModel` in `arcface/recognizer_test.go`.

### Why centerface.onnx/retinaface.onnx aren't straight upstream downloads

- **centerface.onnx**: Star-Clouds' own published file declares a fixed
  `[10,3,32,32]` input (a PyTorch trace artifact with no `dynamic_axes` set)
  that ONNX Runtime enforces strictly, even though `cv2.dnn` -- what
  CenterFace's own reference implementation uses -- tolerates it. This copy
  has only its shape metadata relaxed to dynamic batch/height/width (via
  `onnx.save`, not touching the trained weights) so it works with
  `onnxruntime_go`'s `DynamicAdvancedSession`.
- **retinaface.onnx**: InsightFace/biubug6 only publish PyTorch `.pth`
  weights, no ONNX export at all. Produced with the upstream repo's own
  `convert_to_onnx.py` against their published `Resnet50_Final.pth`
  (state_dict keys matched with zero missing/unexpected, confirming the
  right checkpoint), fixed 640x640 input. The fp32 export was ~109MB, over
  GitHub's 100MB git file-size limit (release assets don't have that
  limit, but the file was already being produced anyway); weights (not
  the float32 input/output tensors) were converted to float16 to bring it
  to ~52MB, re-validated afterward (box/landmark differences from the
  fp32 version are ~0.01px).

Both are MIT (same as their upstream projects) and trained on WIDER FACE, so
hosting modified/re-exported copies on go-onnxface's own
[`models-v1` release](https://github.com/leandroveronezi/go-onnxface/releases/tag/models-v1)
doesn't change anything about their licensing.

## License

[MIT](LICENSE)
