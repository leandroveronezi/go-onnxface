package arcface_test

import (
	"image/jpeg"
	"math"
	"os"
	"testing"

	"github.com/leandroveronezi/go-onnxface"
	"github.com/leandroveronezi/go-onnxface/arcface"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

/*
This package ships no weights and downloads none (see the package doc),
so it can't be validated against a pinned model the way yunet/sface/
centerface are. TestRecognizerAgainstLocalModel instead reads a model
you provide locally, entirely through environment variables, and skips
itself if they're unset -- which is always, in CI. To run it locally
against your own (correctly licensed) weights:

	ONNXFACE_ORT_LIB=/path/to/libonnxruntime.so \
	ARCFACE_TEST_MODEL=/path/to/model.onnx \
	ARCFACE_TEST_INPUT_NAME=input.1 \
	ARCFACE_TEST_OUTPUT_NAME=683 \
	go test ./arcface/...
*/
func TestRecognizerAgainstLocalModel(t *testing.T) {

	ortLib := os.Getenv("ONNXFACE_ORT_LIB")
	modelPath := os.Getenv("ARCFACE_TEST_MODEL")
	inputName := os.Getenv("ARCFACE_TEST_INPUT_NAME")
	outputName := os.Getenv("ARCFACE_TEST_OUTPUT_NAME")
	if ortLib == "" || modelPath == "" || inputName == "" || outputName == "" {
		t.Skip("ONNXFACE_ORT_LIB/ARCFACE_TEST_MODEL/ARCFACE_TEST_INPUT_NAME/ARCFACE_TEST_OUTPUT_NAME not all set, skipping (see the doc comment on this test)")
	}

	if err := onnxface.InitEnvironment(ortLib); err != nil {
		t.Fatalf("InitEnvironment: %v", err)
	}
	defer onnxface.CloseEnvironment()

	det, err := yunet.NewDetector("../models/face_detection_yunet_2023mar.onnx")
	if err != nil {
		t.Fatalf("NewDetector: %v", err)
	}
	defer det.Close()

	rec, err := arcface.NewRecognizer(modelPath, arcface.Config{
		InputName:  inputName,
		OutputName: outputName,
	})
	if err != nil {
		t.Fatalf("NewRecognizer: %v", err)
	}
	defer rec.Close()

	feature := func(path string) []float32 {
		f, err := os.Open(path)
		if err != nil {
			t.Fatalf("Open(%s): %v", path, err)
		}
		defer f.Close()
		img, err := jpeg.Decode(f)
		if err != nil {
			t.Fatalf("jpeg.Decode(%s): %v", path, err)
		}

		faces, err := det.Detect(img)
		if err != nil || len(faces) != 1 {
			t.Fatalf("Detect(%s): %d faces, err=%v", path, len(faces), err)
		}

		aligned := rec.Align(img, faces[0].Landmarks)
		if b := aligned.Bounds(); b.Dx() != 112 || b.Dy() != 112 {
			t.Fatalf("Align(%s) produced %dx%d, want 112x112", path, b.Dx(), b.Dy())
		}

		feat, err := rec.Feature(aligned)
		if err != nil {
			t.Fatalf("Feature(%s): %v", path, err)
		}
		return feat
	}

	amyFeat := feature("../testdata/amy.jpg")
	bernFeat := feature("../testdata/bernadette.jpg")

	cosSame := onnxface.Match(amyFeat, amyFeat, onnxface.DistanceCosine)
	if math.Abs(cosSame-1.0) > 1e-4 {
		t.Errorf("cosine(amy,amy) = %v, want ~1.0", cosSame)
	}

	cosDiff := onnxface.Match(amyFeat, bernFeat, onnxface.DistanceCosine)
	if cosDiff > 0.3 {
		t.Errorf("cosine(amy,bernadette) = %v, expected well below 0.3 for different people", cosDiff)
	}

}
