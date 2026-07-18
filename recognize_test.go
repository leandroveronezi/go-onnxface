package onnxface_test

import (
	"image"
	"image/jpeg"
	"math"
	"os"
	"testing"

	"github.com/leandroveronezi/go-onnxface"
	"github.com/leandroveronezi/go-onnxface/sface"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

const (
	yunetModelPath = "models/face_detection_yunet_2023mar.onnx"
	sfaceModelPath = "models/face_recognition_sface_2021dec.onnx"
)

func ortSharedLibraryPath(t *testing.T) string {
	t.Helper()

	path := os.Getenv("ONNXFACE_ORT_LIB")
	if path == "" {
		t.Skip("ONNXFACE_ORT_LIB not set, skipping test that needs the onnxruntime shared library")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("ONNXFACE_ORT_LIB=%s: %v", path, err)
	}
	return path
}

// initForTest initializes the ONNX Runtime environment and registers a
// cleanup to tear it down. It's idempotent within a single test: calling
// it more than once (e.g. once per Recognizer under test) is safe, since
// only the call that actually initializes registers a matching cleanup.
func initForTest(t *testing.T) {
	t.Helper()

	if onnxface.IsInitialized() {
		return
	}

	path := ortSharedLibraryPath(t)

	if err := onnxface.InitEnvironment(path); err != nil {
		t.Fatalf("InitEnvironment: %v", err)
	}
	t.Cleanup(func() { onnxface.CloseEnvironment() })
}

func loadTestImage(t *testing.T, path string) image.Image {
	t.Helper()

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open(%s): %v", path, err)
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		t.Fatalf("jpeg.Decode(%s): %v", path, err)
	}

	return img
}

/*
TestEngineRecognizeAndCompare is an end-to-end test of the high-level API
(Engine.Recognize + Compare) built on top of the yunet/sface packages
already validated individually against real cv2 reference runs. It checks
that: the same photo compared to itself is an unambiguous IsMatch, a
different person is an unambiguous non-match, and a tolerance below the
observed distance correctly flips a would-be match into a rejection --
this is the "the caller picks the threshold, we don't assume one for
them" contract carried over from go-face/go-recognizer's Tolerance.
*/
func TestEngineRecognizeAndCompare(t *testing.T) {
	initForTest(t)

	det, err := yunet.NewDetector(yunetModelPath)
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}

	rec, err := sface.NewRecognizer(sfaceModelPath)
	if err != nil {
		det.Close()
		t.Fatalf("NewRecognizer: %v", err)
	}

	engine := onnxface.NewEngine(det, rec)
	defer engine.Close()

	amyImg := loadTestImage(t, "testdata/amy.jpg")
	bernImg := loadTestImage(t, "testdata/bernadette.jpg")

	amyResults, err := engine.Recognize(amyImg)
	if err != nil {
		t.Fatalf("Recognize(amy): %v", err)
	}
	if len(amyResults) != 1 {
		t.Fatalf("Recognize(amy): got %d results, want 1", len(amyResults))
	}

	bernResults, err := engine.Recognize(bernImg)
	if err != nil {
		t.Fatalf("Recognize(bernadette): %v", err)
	}
	if len(bernResults) != 1 {
		t.Fatalf("Recognize(bernadette): got %d results, want 1", len(bernResults))
	}

	amyFeat := amyResults[0].Feature
	bernFeat := bernResults[0].Feature

	const tolerance = 1.128 // OpenCV's suggested SFace L2 starting point

	same := onnxface.Compare(amyFeat, amyFeat, tolerance)
	if !same.IsMatch {
		t.Errorf("Compare(amy,amy) = %+v, want IsMatch=true", same)
	}
	if same.Distance > 1e-4 {
		t.Errorf("Compare(amy,amy).Distance = %v, want ~0", same.Distance)
	}
	if math.Abs(same.Confidence-1.0) > 1e-4 {
		t.Errorf("Compare(amy,amy).Confidence = %v, want ~1.0", same.Confidence)
	}

	diff := onnxface.Compare(amyFeat, bernFeat, tolerance)
	if diff.IsMatch {
		t.Errorf("Compare(amy,bernadette) = %+v, want IsMatch=false", diff)
	}
	if math.Abs(diff.Distance-1.325) > 0.1 {
		t.Errorf("Compare(amy,bernadette).Distance = %v, want ~1.325 (within 0.1)", diff.Distance)
	}

	// A tolerance set above the observed distance must accept the same
	// pair that a stricter one rejects -- confirms the caller's threshold
	// choice actually drives the decision, not a hardcoded one.
	lenient := onnxface.Compare(amyFeat, bernFeat, diff.Distance+0.5)
	if !lenient.IsMatch {
		t.Errorf("Compare(amy,bernadette) with a lenient tolerance = %+v, want IsMatch=true", lenient)
	}
}
