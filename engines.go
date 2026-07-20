package onnxface

import (
	"fmt"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface/arcface"
	"github.com/leandroveronezi/go-onnxface/centerface"
	"github.com/leandroveronezi/go-onnxface/face"
	"github.com/leandroveronezi/go-onnxface/ghostface"
	"github.com/leandroveronezi/go-onnxface/liveness"
	"github.com/leandroveronezi/go-onnxface/retinaface"
	"github.com/leandroveronezi/go-onnxface/seetaface6"
	"github.com/leandroveronezi/go-onnxface/sface"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

// DetectorEngine selects which detection engine Recognizer.Init loads.
type DetectorEngine int

const (
	// DetectorYuNet is the default: smallest model, fixed 640x640 input.
	DetectorYuNet DetectorEngine = iota
	// DetectorCenterFace trades some latency for better recall.
	DetectorCenterFace
	// DetectorRetinaFace is the heaviest and most accurate of the three
	// -- see the README's Benchmarks section for the numbers.
	DetectorRetinaFace
)

// RecognizerEngine selects which recognition engine Recognizer.Init loads.
type RecognizerEngine int

const (
	// RecognizerSFace is the default: the only recognition model with an
	// explicit commercial grant on its published weights -- see the
	// README's Licensing section.
	RecognizerSFace RecognizerEngine = iota
	// RecognizerArcFace requires Config.RecognizerFile and Config.ArcFace
	// (no bundled weights -- see the arcface package doc for licensing).
	RecognizerArcFace
	// RecognizerGhostFace requires Config.RecognizerFile (no bundled
	// weights -- see the ghostface package doc for licensing).
	RecognizerGhostFace
)

// LivenessEngine selects which liveness (print/replay spoof detection)
// engine Recognizer.Init loads, if any.
type LivenessEngine int

const (
	// LivenessNone (the default) loads no liveness engine -- CheckLiveness
	// returns ErrNoLivenessEngine until Config.Liveness is set and Init
	// is called again.
	LivenessNone LivenessEngine = iota
	// LivenessMiniFAS is the MiniFASNetV2/V1SE ensemble (the liveness
	// package). Balanced trade-off -- see the go-onnxface-benchmarks
	// README for the numbers this and LivenessSeetaFace6 were compared
	// against.
	LivenessMiniFAS
	// LivenessSeetaFace6 (the seetaface6 package) catches print/replay
	// spoofs far more aggressively than LivenessMiniFAS, at the cost of
	// rejecting far more real people -- a real precision/recall
	// trade-off, not a strict improvement.
	LivenessSeetaFace6
)

func newDetectorEngine(e DetectorEngine, dir, file string) (face.FaceDetector, error) {
	switch e {
	case DetectorCenterFace:
		if file == "" {
			file = centerface.ModelFileName
		}
		return centerface.NewDetector(filepath.Join(dir, file))
	case DetectorRetinaFace:
		if file == "" {
			file = retinaface.ModelFileName
		}
		return retinaface.NewDetector(filepath.Join(dir, file))
	default:
		if file == "" {
			file = defaultDetectorModel
		}
		return yunet.NewDetector(filepath.Join(dir, file))
	}
}

func newRecognizerEngine(e RecognizerEngine, dir, file string, arcCfg arcface.Config) (face.FaceRecognizer, error) {
	switch e {
	case RecognizerArcFace:
		if file == "" {
			return nil, fmt.Errorf("onnxface: RecognizerArcFace requires Config.RecognizerFile -- this package ships no weights, see the arcface package doc")
		}
		return arcface.NewRecognizer(filepath.Join(dir, file), arcCfg)
	case RecognizerGhostFace:
		if file == "" {
			return nil, fmt.Errorf("onnxface: RecognizerGhostFace requires Config.RecognizerFile -- this package ships no weights, see the ghostface package doc")
		}
		return ghostface.NewRecognizer(filepath.Join(dir, file))
	default:
		if file == "" {
			file = defaultRecognizerModel
		}
		return sface.NewRecognizer(filepath.Join(dir, file))
	}
}

func newLivenessEngine(e LivenessEngine, dir string, files [2]string) (face.LivenessDetector, error) {
	switch e {
	case LivenessSeetaFace6:
		f1, f2 := files[0], files[1]
		if f1 == "" {
			f1 = seetaface6.FasFirstFileName
		}
		if f2 == "" {
			f2 = seetaface6.FasSecondFileName
		}
		return seetaface6.NewDetector(filepath.Join(dir, f1), filepath.Join(dir, f2))
	default: // LivenessMiniFAS
		f1, f2 := files[0], files[1]
		if f1 == "" {
			f1 = liveness.ModelV2FileName
		}
		if f2 == "" {
			f2 = liveness.ModelV1SEFileName
		}
		return liveness.NewDetector(filepath.Join(dir, f1), filepath.Join(dir, f2))
	}
}
