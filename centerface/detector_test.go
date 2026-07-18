package centerface_test

import (
	"image"
	"image/jpeg"
	"math"
	"os"
	"testing"

	"github.com/leandroveronezi/go-onnxface/centerface"
	"github.com/leandroveronezi/go-onnxface/face"
)

const centerfaceModelPath = "../models/centerface.onnx"

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
the Go implementation against a real cv2.dnn run of CenterFace's own
reference algorithm (centerface.py's decode/nms, ported line-by-line) on
the same image: it reported box (295.18,51.77,133.76,186.37), score
0.8980, and specific landmark coordinates for testdata/amy.jpg. This
catches the exact class of preprocessing-mismatch bug (wrong resize,
wrong channel order, wrong scale/offset decode) that silently produces a
plausible-looking but wrong result instead of an error.
*/
func TestDetectorFindsFaceMatchingPythonReference(t *testing.T) {
	initForTest(t)

	det, err := centerface.NewDetector(centerfaceModelPath)
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

	wantBox := [4]float64{295.18, 51.77, 133.76, 186.37} // x, y, w, h
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

	if math.Abs(float64(f.Score)-0.8980) > 0.01 {
		t.Errorf("Score = %v, want ~0.8980", f.Score)
	}

	wantLandmarks := [5][2]float64{
		{319.97, 120.28},
		{385.26, 116.33},
		{348.12, 150.68},
		{326.91, 185.58},
		{383.37, 181.79},
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
