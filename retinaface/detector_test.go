package retinaface_test

import (
	"image"
	"image/jpeg"
	"math"
	"os"
	"testing"

	"github.com/leandroveronezi/go-onnxface/face"
	"github.com/leandroveronezi/go-onnxface/retinaface"
)

const retinafaceModelPath = "../models/retinaface.onnx"

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
TestDetectorFindsFaceMatchingPythonReference is a regression test pinning
the Go implementation against a real onnxruntime run of RetinaFace's own
reference algorithm (biubug6/Pytorch_Retinaface's PriorBox/decode/NMS,
ported line-by-line) on the same image: it reported box
(295.87,56.97,132.33,179.85), score 0.9998, and specific landmark
coordinates for testdata/amy.jpg. This catches the exact class of
preprocessing-mismatch bug (wrong anchor generation, wrong channel order,
wrong variance decode) that silently produces a plausible-looking but
wrong result instead of an error.
*/
func TestDetectorFindsFaceMatchingPythonReference(t *testing.T) {
	initForTest(t)

	det, err := retinaface.NewDetector(retinafaceModelPath)
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}
	defer det.Close()

	img := loadTestImage(t, "../testdata/amy.jpg")

	faces, err := det.Detect(img)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(faces) != 1 {
		t.Fatalf("got %d faces, want 1", len(faces))
	}

	f := faces[0]

	wantBox := [4]float64{295.87, 56.97, 132.33, 179.85} // x, y, w, h
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

	if math.Abs(float64(f.Score)-0.9998) > 0.01 {
		t.Errorf("Score = %v, want ~0.9998", f.Score)
	}

	wantLandmarks := [5][2]float64{
		{321.69, 113.19},
		{379.34, 109.17},
		{345.58, 148.39},
		{326.10, 178.25},
		{384.44, 174.57},
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
