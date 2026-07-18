package sface

import (
	"image"
	"image/jpeg"
	"math"
	"os"
	"testing"

	"github.com/leandroveronezi/go-onnxface"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

const (
	yunetModelPath = "../models/face_detection_yunet_2023mar.onnx"
	sfaceModelPath = "../models/face_recognition_sface_2021dec.onnx"
)

// ortSharedLibraryPath reads the onnxruntime shared library path from the
// ONNXFACE_ORT_LIB environment variable, skipping the test if it's unset
// (the library is a large platform-specific binary, not checked in).
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

	if err := onnxface.Init(path); err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(func() { onnxface.Close() })
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

func featureFor(t *testing.T, det *yunet.Detector, rec *Recognizer, imgPath string) []float32 {
	t.Helper()

	img := loadTestImage(t, imgPath)

	faces, err := det.Detect(img)
	if err != nil {
		t.Fatalf("Detect(%s): %v", imgPath, err)
	}
	if len(faces) != 1 {
		t.Fatalf("Detect(%s): got %d faces, want 1", imgPath, len(faces))
	}

	aligned := AlignCrop(img, faces[0].Landmarks)

	feat, err := rec.Feature(aligned)
	if err != nil {
		t.Fatalf("Feature(%s): %v", imgPath, err)
	}
	return feat
}

// TestRecognizerMatchesPythonReference is a regression test pinning the Go
// alignCrop+feature+match pipeline against a real cv2.FaceRecognizerSF run
// (same algorithm this package's AlignCrop/Feature were ported from) on
// testdata/amy.jpg and testdata/bernadette.jpg, both detected on the same
// 640x640 letterboxed canvas the yunet package's Detector uses:
// opencv-python reported cosine(amy,amy)=1.0, cosine(amy,bernadette)=0.1217,
// l2(amy,amy)=0.0, l2(amy,bernadette)=1.325. This catches the exact class of
// alignment/preprocessing-mismatch bug (wrong similarity transform, wrong
// channel order, wrong normalization) that silently produces a
// plausible-looking but wrong embedding instead of an error.
func TestRecognizerMatchesPythonReference(t *testing.T) {
	initForTest(t)

	det, err := yunet.NewDetector(yunetModelPath)
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}
	defer det.Close()

	rec, err := NewRecognizer(sfaceModelPath)
	if err != nil {
		t.Fatalf("NewRecognizer: %v", err)
	}
	defer rec.Close()

	amyFeat := featureFor(t, det, rec, "../testdata/amy.jpg")
	bernFeat := featureFor(t, det, rec, "../testdata/bernadette.jpg")

	if len(amyFeat) != featureSize {
		t.Fatalf("len(amyFeat) = %d, want %d", len(amyFeat), featureSize)
	}

	cosSame := onnxface.Match(amyFeat, amyFeat, onnxface.DistanceCosine)
	if math.Abs(cosSame-1.0) > 1e-4 {
		t.Errorf("cosine(amy,amy) = %v, want ~1.0", cosSame)
	}

	cosDiff := onnxface.Match(amyFeat, bernFeat, onnxface.DistanceCosine)
	if math.Abs(cosDiff-0.1217) > 0.05 {
		t.Errorf("cosine(amy,bernadette) = %v, want ~0.1217 (within 0.05)", cosDiff)
	}
	// OpenCV's own SFace docs recommend ~0.363 as the cosine "same person"
	// threshold; two different people must land well below it.
	if cosDiff > 0.3 {
		t.Errorf("cosine(amy,bernadette) = %v, expected well below the ~0.363 same-person threshold", cosDiff)
	}

	l2Same := onnxface.Match(amyFeat, amyFeat, onnxface.DistanceL2)
	if l2Same > 1e-4 {
		t.Errorf("l2(amy,amy) = %v, want ~0.0", l2Same)
	}

	l2Diff := onnxface.Match(amyFeat, bernFeat, onnxface.DistanceL2)
	if math.Abs(l2Diff-1.325) > 0.1 {
		t.Errorf("l2(amy,bernadette) = %v, want ~1.325 (within 0.1)", l2Diff)
	}
}
