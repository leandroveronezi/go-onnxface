package seetaface6_test

import (
	"image"
	"image/jpeg"
	"os"
	"testing"

	"github.com/leandroveronezi/go-onnxface/face"
	"github.com/leandroveronezi/go-onnxface/seetaface6"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

const (
	yunetModelPath     = "../models/face_detection_yunet_2023mar.onnx"
	fasFirstModelPath  = "../models/fas_first.onnx"
	fasSecondModelPath = "../models/fas_second.onnx"
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

func initForTest(t *testing.T) {
	t.Helper()

	path := ortSharedLibraryPath(t)

	if err := face.InitEnvironment(path); err != nil {
		t.Fatalf("InitEnvironment: %v", err)
	}
	t.Cleanup(func() { face.CloseEnvironment() })
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
TestDetectorRealPhoto is a smoke test against testdata/amy.jpg (a real
photo, no print/replay spoof involved) -- confirms the fas_second
full-image gate doesn't fire and fas_first's fused decision lands on
"live", the same way liveness.Detector's own equivalent test does.
*/
func TestDetectorRealPhoto(t *testing.T) {
	initForTest(t)

	yn, err := yunet.NewDetector(yunetModelPath)
	if err != nil {
		t.Fatalf("yunet.NewDetector: %v", err)
	}
	defer yn.Close()

	img := loadTestImage(t, "../testdata/amy.jpg")

	faces, err := yn.Detect(img)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(faces) != 1 {
		t.Fatalf("got %d faces, want 1", len(faces))
	}

	det, err := seetaface6.NewDetector(fasFirstModelPath, fasSecondModelPath)
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}
	defer det.Close()

	result, err := det.Detect(img, faces[0])
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}

	if !result.IsLive {
		t.Errorf("IsLive = false, want true (testdata/amy.jpg is a real photo)")
	}
	if result.Score < 0.8 {
		t.Errorf("Score = %v, want >= 0.8 (fuseThreshold)", result.Score)
	}
}
