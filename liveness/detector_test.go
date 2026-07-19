package liveness_test

import (
	"image"
	"image/jpeg"
	"math"
	"os"
	"testing"

	"github.com/leandroveronezi/go-onnxface/face"
	"github.com/leandroveronezi/go-onnxface/liveness"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

const (
	yunetModelPath = "../models/face_detection_yunet_2023mar.onnx"
	v2ModelPath    = "../models/minifasnet_v2.onnx"
	v1seModelPath  = "../models/minifasnet_v1se.onnx"
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
TestDetectorMatchesPythonReference is a regression test pinning the Go
implementation against a real onnxruntime run of
Silent-Face-Anti-Spoofing's own ensemble algorithm (CropImage._get_new_box,
its deliberately-not-normalized ToTensor, softmax + sum + argmax/len
decision, all ported line-by-line) on testdata/amy.jpg using the same
box yunet.Detector itself produces: opencv-python's onnxruntime reported
label=1 (real), score=0.9947. This catches the exact class of
preprocessing-mismatch bug (wrong crop box, wrong normalization, wrong
ensemble math) that silently produces a plausible-looking but wrong
result instead of an error.
*/
func TestDetectorMatchesPythonReference(t *testing.T) {
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

	det, err := liveness.NewDetector(v2ModelPath, v1seModelPath)
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}
	defer det.Close()

	result, err := det.Detect(img, faces[0].Rectangle)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}

	if !result.IsLive {
		t.Errorf("IsLive = false, want true (testdata/amy.jpg is a real photo)")
	}
	if math.Abs(result.Score-0.9947) > 0.02 {
		t.Errorf("Score = %v, want ~0.9947 (within 0.02)", result.Score)
	}
}
