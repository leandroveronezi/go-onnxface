## go-onnxface

[![CI](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml/badge.svg)](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/leandroveronezi/go-onnxface.svg)](https://pkg.go.dev/github.com/leandroveronezi/go-onnxface)
![MIT Licensed](https://img.shields.io/github/license/leandroveronezi/go-onnxface.svg)

🇧🇷 [Leia em português](README.pt-BR.md)

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
| Detection | `retinaface` | [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) (resnet50) | MIT | Heaviest of the three (~52MB float16 vs CenterFace's ~7.5MB/YuNet's ~230KB). Fixed 640x640 input (letterboxed). |
| Recognition | `sface` | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache-2.0 | Only recognition model found so far with an explicit commercial grant on the weights -- see Licensing below. |
| Recognition | `arcface` | any ArcFace-family ONNX export | *depends on your weights* | A bridge, not a model: ships no weights, downloads none. See Licensing below before using it. |
| Recognition | `ghostface` | [GhostFaceNetV1](https://github.com/HamadYA/GhostFaceNets) | *depends on your weights* | Same situation as `arcface` (MS1MV2/MS1MV3-trained) -- a bridge, no bundled/downloaded weights. Modern (2023) and competitive with ArcFace, unlike the other DeepFace-wrapped recognition models (VGG-Face/OpenFace/2014-era "DeepFace"), which are old enough that adding them wouldn't beat what's already here. |
| Liveness | `liveness` | [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (MiniFASNetV2 + MiniFASNetV1SE) | Apache-2.0 | Default liveness engine. Print/replay spoof detection -- trained for this task specifically, not a face-identity dataset, so none of the recognition-model licensing caveats apply. |
| Liveness | `seetaface6` | [SeetaFace6](https://github.com/seetafaceengine/SeetaFace6) (fas_first + fas_second) | BSD | Catches print/replay spoofs far more aggressively than `liveness`, at the cost of rejecting far more real people -- a real precision/recall trade-off, not a strict improvement. See the go-onnxface-benchmarks README for the numbers this was validated against. |

Both `liveness` and `seetaface6` implement the shared `face.LivenessDetector`
contract, so they're interchangeable via `Recognizer.Model.Liveness` (see
Usage below) or directly, same as the detection/recognition engines.

See [Benchmarks](#benchmarks) below for accuracy/latency numbers per package.

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
- ✅ Liveness/anti-spoof (`liveness.Detector`) -- the crop/preprocessing/ensemble
  math ported line-by-line from Silent-Face-Anti-Spoofing's own source (not just
  its README), validated against a real onnxruntime run of the unmodified
  original models on `testdata/amy.jpg` (a real photo, correctly scored
  live, ~0.99).
- ✅ Liveness/anti-spoof (`seetaface6.Detector`) -- the fas_first/fas_second
  fusion (SSD full-image gate, YCrCb-converted aligned-crop classifier) ported
  line-by-line from SeetaFace6's own C++ source
  (`FaceAntiSpoofingX/src/seeta/FaceAntiSpoofing.cpp`), validated against a real
  onnxruntime run on `testdata/amy.jpg` (correctly scored live, ~0.999).
- ✅ `ghostface` bridge (code only, bring your own weights, same reasoning as
  `arcface`) -- validated locally against a real GhostFaceNetV1 run, converted
  from the original Keras weights via tf2onnx (same-person cosine ~1.0,
  different-person cosine ~0.008, using the exact landmarks the Go side
  itself produced -- confirms the alignment/preprocessing port is correct,
  not just "close enough").

### Benchmarks

Measured against uncurated, real-world datasets -- chosen specifically because they
include real pose/lighting/occlusion variation instead of posed studio photos:

- **Detection**: [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) validation set --
  3,226 images, 19,926 ground-truth faces ≥20px tall. Metric: recall at IoU≥0.5,
  every difficulty level combined. This is *not* the official WIDER FACE eval-tools
  protocol (separate easy/medium/hard subsets, PASCAL-VOC-style average precision),
  so treat the two columns below as directionally comparable, not the same metric.
- **Recognition**: [CFP-FP](http://www.cfpw.io/) (Celebrities in Frontal-Profile) --
  7,000 frontal-vs-profile verification pairs across all 10 folds -- and
  [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/) -- 7,000 pairs generated locally
  (same/different identity, fixed seed) from a 567-identity/16,488-image repackaging,
  since AgeDB doesn't ship a fixed protocol the way CFP-FP does. Metric for both:
  accuracy at the single best-achievable threshold, swept post-hoc over every pair --
  slightly more optimistic than the standard 10-fold cross-validated-threshold
  protocol most papers report.
- **Liveness**: [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) test
  split -- ~6,700 images (1 of 10 public shards), roughly 30%/70% live/spoof.

**Detection (WIDER FACE):**

| Package | Recall | Avg latency/image (CPU, see note) | Published reference (original model/paper) |
|---------|--------|-------------------------------------|----------------------------------------------|
| `yunet` | 70.67% | 36.8ms | 88.44% / 86.56% / 75.03% easy/medium/hard AP ([opencv_zoo](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)) |
| `centerface` | 78.92% | 247.5ms | 92.2% / 91.1% / 78.2% easy/medium/hard mAP, single-scale ([upstream](https://github.com/Star-Clouds/CenterFace)) |
| `retinaface` | 76.55% | 384.9ms | 96.5% / 95.6% / 90.4% easy/medium/hard mAP ([paper](https://arxiv.org/abs/1905.00641)) |

**Recognition (CFP-FP and AgeDB, one column each):**

| Package | CFP-FP accuracy | AgeDB accuracy | Avg latency/image (CPU) | Published reference |
|---------|------------------|-----------------|---------------------------|----------------------|
| `sface` | 97.13% | 95.47% | 15.5ms | 95.26% CFP-FP ([paper](https://arxiv.org/abs/2205.12010), ResNet50/CASIA-WebFace config -- the shipped weights are a lighter MobileFaceNet, not necessarily identical) |
| `arcface` (buffalo_l) | 99.51% | 97.78% | 112.9ms | 99.33% CFP-FP ([InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)) |
| `ghostface` | 96.80% | 96.48% | 13.7ms | 96.83% CFP-FP ([paper](https://www.researchgate.net/publication/369930264_GhostFaceNets_Lightweight_Face_Recognition_Model_from_Cheap_Operations)) |

**Liveness (CelebA-Spoof):**

| Package | Live accuracy | Spoof accuracy | Avg latency/image (CPU) |
|---------|----------------|------------------|---------------------------|
| `liveness` | 74.18% | 69.67% | ~12ms |

Latency is CPU-only inference (no GPU), measured on an Intel i7-1165G7
(4 cores/8 threads) -- treat as directional for your own hardware, not an
absolute number. AgeDB is a harder benchmark for every recognizer than CFP-FP (it
stresses age variation, not just frontal-vs-profile pose) -- `arcface` drops the
most (99.51%->97.78%), suggesting it generalizes across age somewhat worse than the
others. `liveness`'s CelebA-Spoof numbers are noticeably lower than what a
production classroom-attendance photo would produce (see the print/replay-specific
validation already described above) -- CelebA-Spoof's images are lower-resolution,
surveillance-camera-style captures that stress the model harder than a phone/webcam
photo does.

Pose is consistently the hardest attribute for all three detectors: WIDER FACE's own
atypical-pose faces drop recall from ~73-81% (typical pose) down to ~38-58%
(atypical) -- a bigger gap than blur or occlusion produce on their own. RetinaFace is
the most robust to atypical pose specifically (57.68% vs CenterFace's 53.15% and
YuNet's 38.37%); CenterFace has the best overall recall and the best occlusion/
illumination handling. None of the three -- or any face detector -- can find a face
that isn't visible at all (e.g. someone looking straight down): that's a person/head
detection problem, not face detection, and out of scope here.

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

The `arcface`/`ghostface` packages exist for models in that family anyway, but
deliberately as *just code*: neither ships or downloads any `.onnx` file, because
doing so would mean this MIT-licensed repository redistributing someone else's
non-commercial-only weights -- the wrapper's own license doesn't change what
license the weights carry. Bring your own file, and make sure you have the rights
to it for how you intend to use it: research use of InsightFace's published
`buffalo_l`/`antelopev2` is generally fine under their terms, commercial use needs
their paid license (see their commercial licensing page). GhostFaceNets' own
MS1MV2/MS1MV3-trained weights carry the same research-only restriction, but --
unlike InsightFace -- HamadYA doesn't offer a paid commercial license at all, so
commercial use of that specific weight file isn't something you can currently buy
your way into; a from-scratch retrain on your own licensed data would be the only
route. This isn't legal advice, just how the ecosystem's licensing actually works
in practice -- when unsure, ask a lawyer, not this README.

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

result, err := rec.Identify("photo.jpg")
switch {
case errors.Is(err, onnxface.ErrNoFace), errors.Is(err, onnxface.ErrMultipleFaces):
    // photo.jpg doesn't have exactly one face
case errors.Is(err, onnxface.ErrNoMatch):
    // no Dataset entry within rec.Tolerance
case err != nil:
    // something else went wrong (I/O, decode, ...)
}
fmt.Println(result.Id, result.Distance, result.Confidence)
```

`AddImageToDataset`/`RecognizeSingle`/`Identify` return these sentinel
errors for their "expected" failure conditions -- check with `errors.Is`
instead of matching on the error message text, which isn't part of the
API contract and may change. Any other error (I/O, image decoding, etc.)
is wrapped with `%w`, so `errors.Unwrap`/`errors.As` still reach the
underlying cause.

By default `Recognizer` uses `yunet`+`sface` and no liveness engine --
set `Model` before `Init` to pick different ones (any combination of the
Engines table above), plus any per-engine file/config overrides:

```go
rec := &onnxface.Recognizer{}
rec.Model.Detector = onnxface.DetectorRetinaFace  // more accurate, slower -- see Benchmarks
rec.Model.Liveness = onnxface.LivenessSeetaFace6  // adds CheckLiveness below

if err := rec.DownloadModels("models"); err != nil { // fetches only what Model selects
    // ...
}
if err := rec.Init("models"); err != nil {
    // ...
}
defer rec.Close()

liveness, err := rec.CheckLiveness("photo.jpg")
switch {
case errors.Is(err, onnxface.ErrNoLivenessEngine):
    // Model.Liveness was left at the default (LivenessNone)
case err != nil:
    // photo.jpg doesn't have exactly one face, or something else went wrong
default:
    fmt.Println(liveness.IsLive, liveness.Score)
}
```

`RecognizerArcFace`/`RecognizerGhostFace` ship no weights (see Licensing
above), so `Model.RecognizerFile` and, for `arcface`, `Model.ArcFace` are
required -- `Init` returns a clear error otherwise, and `DownloadModels`
is a no-op for either.

`Recognizer.Tolerance` defaults to 1.128 (OpenCV's suggested SFace L2
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

`ghostface` is a narrower bridge (one specific architecture, GhostFaceNetV1, not
"any ONNX export of this family" like `arcface`), so it doesn't need a `Config` --
just point it at your own conversion (see the package doc for the exact recipe):

```go
rec, _ := ghostface.NewRecognizer("/path/to/your/ghostfacenet.onnx")
```

Liveness detection takes a face rectangle from any detector (it isn't tied to
`face.FaceDetector`) and classifies it as a live face or a print/replay spoof:

```go
import "github.com/leandroveronezi/go-onnxface/liveness"

liveness.DownloadModel("models") // both ensemble models together -- always used as a pair

live, _ := liveness.NewDetector("models/minifasnet_v2.onnx", "models/minifasnet_v1se.onnx")
defer live.Close()

faces, _ := det.Detect(img) // any FaceDetector
result, _ := live.Detect(img, faces[0].Rectangle)
fmt.Println(result.IsLive, result.Score)
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
curl -sL -o models/minifasnet_v2.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/minifasnet_v2.onnx
curl -sL -o models/minifasnet_v1se.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/minifasnet_v1se.onnx

go test ./...
```

Tests that need the onnxruntime shared library or a model file skip
themselves if it/they aren't present, so a partial fetch just skips fewer
tests rather than failing the ones that don't need what's missing.
`TestDownloadModels`/`centerface.TestDownloadModel`/`retinaface.TestDownloadModel`/
`liveness.TestDownloadModel` additionally do a real network download into a
fresh directory each, to verify the download path itself (not just a
pre-fetched file) works end to end. `arcface`'s test needs its own
locally-provided (correctly licensed) model and skips itself without it --
see the doc comment on `TestRecognizerAgainstLocalModel` in
`arcface/recognizer_test.go`.

### Why centerface.onnx/retinaface.onnx/minifasnet_*.onnx aren't straight upstream downloads

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
- **minifasnet_v2.onnx/minifasnet_v1se.onnx**: same story as retinaface.onnx --
  minivision-ai only publish PyTorch `.pth` weights. Produced with the
  upstream repo's own model definitions (`src/model_lib/MiniFASNet.py`)
  against their published `2.7_80x80_MiniFASNetV2.pth`/
  `4_0_0_80x80_MiniFASNetV1SE.pth`, fixed 80x80 input, small enough
  (~1.7MB each) that no float16 conversion was needed.

All are MIT/Apache-2.0 (same as their upstream projects) and trained on
tasks with no identity-labeled data (WIDER FACE bounding boxes, or
Silent-Face-Anti-Spoofing's own live/spoof data), so hosting modified/
re-exported copies on go-onnxface's own
[`models-v1` release](https://github.com/leandroveronezi/go-onnxface/releases/tag/models-v1)
doesn't change anything about their licensing.

## License

[MIT](LICENSE)
