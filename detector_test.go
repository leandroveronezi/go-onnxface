package onnxface

import (
	"image"
	"image/jpeg"
	"math"
	"os"
	"testing"
)

const yunetModelPath = "models/face_detection_yunet_2023mar.onnx"

// initForTest initializes the ONNX Runtime environment and registers a
// cleanup to tear it down, so each test that needs it can call this
// independently without conflicting with others (tests run sequentially
// within a package by default).
func initForTest(t *testing.T) {
	t.Helper()

	path := ortSharedLibraryPath(t)

	if err := Init(path); err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(func() { Close() })
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

// TestDetectorFindsFaceMatchingPythonReference is a regression test
// pinning the Go implementation against a real cv2.FaceDetectorYN run
// (the same C++ algorithm this package's pre/post-processing was ported
// from) on the same image: opencv-python reported box
// (295.8,45.1,129.1,192.5), score 0.9492, and specific landmark
// coordinates for testdata/amy.jpg. This catches the exact class of
// preprocessing-mismatch bug (wrong normalization, wrong channel order,
// wrong anchor decode) that silently produces plausible-looking but
// wrong results instead of an error.
func TestDetectorFindsFaceMatchingPythonReference(t *testing.T) {
	initForTest(t)

	det, err := NewDetector(yunetModelPath)
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}
	defer det.Close()

	img := loadTestImage(t, "testdata/amy.jpg")

	faces, err := det.Detect(img)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(faces) != 1 {
		t.Fatalf("got %d faces, want 1", len(faces))
	}

	f := faces[0]

	wantBox := [4]float64{295.8, 45.1, 129.1, 192.5} // x, y, w, h
	gotBox := [4]float64{
		float64(f.Rectangle.Min.X),
		float64(f.Rectangle.Min.Y),
		float64(f.Rectangle.Dx()),
		float64(f.Rectangle.Dy()),
	}
	for i, want := range wantBox {
		if math.Abs(gotBox[i]-want) > 2 {
			t.Errorf("box[%d] = %v, want %v (within 2px)", i, gotBox[i], want)
		}
	}

	if math.Abs(float64(f.Score)-0.9492) > 0.01 {
		t.Errorf("Score = %v, want ~0.9492", f.Score)
	}

	wantLandmarks := [5][2]float64{
		{320.7, 110.6},
		{384.2, 111.6},
		{344.4, 149.5},
		{323.4, 179.7},
		{380.3, 180.0},
	}
	for i, want := range wantLandmarks {
		got := f.Landmarks[i]
		dx := float64(got.X) - want[0]
		dy := float64(got.Y) - want[1]
		if math.Abs(dx) > 2 || math.Abs(dy) > 2 {
			t.Errorf("Landmarks[%d] = %v, want ~%v (within 2px)", i, got, want)
		}
	}
}
