package onnxface_test

import (
	"errors"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"

	"github.com/leandroveronezi/go-onnxface"
)

// newTestRecognizer builds a Recognizer against the already-committed
// models/ directory, reusing initForTest's onnxruntime environment setup
// (from recognize_test.go) instead of Init's own auto-download path --
// Recognizer.Init skips that step whenever the environment is already
// initialized, which is exactly what happens here.
func newTestRecognizer(t *testing.T) *onnxface.Recognizer {
	t.Helper()

	initForTest(t)

	rec := &onnxface.Recognizer{}
	if err := rec.Init("models"); err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(rec.Close)

	return rec
}

func TestRecognizerIdentify(t *testing.T) {
	rec := newTestRecognizer(t)

	if err := rec.AddImageToDataset("testdata/amy.jpg", "Amy"); err != nil {
		t.Fatalf("AddImageToDataset: %v", err)
	}

	result, err := rec.Identify("testdata/amy.jpg")
	if err != nil {
		t.Fatalf("Identify(amy): %v", err)
	}
	if result.Id != "Amy" {
		t.Errorf("Id = %q, want Amy", result.Id)
	}
	if result.Distance > 1e-4 {
		t.Errorf("Distance = %v, want ~0", result.Distance)
	}
	if result.Confidence < 0.99 {
		t.Errorf("Confidence = %v, want ~1.0", result.Confidence)
	}

	if _, err := rec.Identify("testdata/bernadette.jpg"); !errors.Is(err, onnxface.ErrNoMatch) {
		t.Errorf("Identify(bernadette): err = %v, want ErrNoMatch", err)
	}
}

func TestRecognizerAddImageToDatasetNoFace(t *testing.T) {
	rec := newTestRecognizer(t)

	blankPath := filepath.Join(t.TempDir(), "blank.jpg")
	writeBlankJPEG(t, blankPath)

	if err := rec.AddImageToDataset(blankPath, "NoOne"); !errors.Is(err, onnxface.ErrNoFace) {
		t.Fatalf("AddImageToDataset(blank): err = %v, want ErrNoFace", err)
	}
}

func TestRecognizerAddImageToDatasetRejectsMultipleFaces(t *testing.T) {
	rec := newTestRecognizer(t)

	if err := rec.AddImageToDataset("examples/fotos/elenco3.jpg", "Group"); !errors.Is(err, onnxface.ErrMultipleFaces) {
		t.Fatalf("AddImageToDataset(elenco3): err = %v, want ErrMultipleFaces", err)
	}
}

// writeBlankJPEG writes a solid-color JPEG with no detectable face to path.
func writeBlankJPEG(t *testing.T, path string) {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 32, 32))

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("Create(%s): %v", path, err)
	}
	defer f.Close()

	if err := jpeg.Encode(f, img, nil); err != nil {
		t.Fatalf("jpeg.Encode: %v", err)
	}
}

func TestRecognizerIdentifyMultiples(t *testing.T) {
	rec := newTestRecognizer(t)

	if err := rec.AddImageToDataset("testdata/amy.jpg", "Amy"); err != nil {
		t.Fatalf("AddImageToDataset: %v", err)
	}

	results, err := rec.IdentifyMultiples("testdata/amy.jpg")
	if err != nil {
		t.Fatalf("IdentifyMultiples: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0].Id != "Amy" {
		t.Errorf("Id = %q, want Amy", results[0].Id)
	}

	results, err = rec.IdentifyMultiples("testdata/bernadette.jpg")
	if err != nil {
		t.Fatalf("IdentifyMultiples(bernadette): %v", err)
	}
	if len(results) != 0 {
		t.Errorf("got %d results, want 0 (no Dataset match within Tolerance)", len(results))
	}
}

func TestRecognizerRecognizeSingleAndMultiples(t *testing.T) {
	rec := newTestRecognizer(t)

	single, err := rec.RecognizeSingle("testdata/amy.jpg")
	if err != nil {
		t.Fatalf("RecognizeSingle: %v", err)
	}
	if single.Score <= 0 {
		t.Errorf("Score = %v, want > 0", single.Score)
	}

	multi, err := rec.RecognizeMultiples("testdata/amy.jpg")
	if err != nil {
		t.Fatalf("RecognizeMultiples: %v", err)
	}
	if len(multi) != 1 {
		t.Fatalf("got %d faces, want 1", len(multi))
	}
}

func TestRecognizerSaveLoadDataset(t *testing.T) {
	rec := newTestRecognizer(t)

	if err := rec.AddImageToDataset("testdata/amy.jpg", "Amy"); err != nil {
		t.Fatalf("AddImageToDataset: %v", err)
	}

	path := filepath.Join(t.TempDir(), "dataset.json")
	if err := rec.SaveDataset(path); err != nil {
		t.Fatalf("SaveDataset: %v", err)
	}

	loaded := newTestRecognizer(t)
	if err := loaded.LoadDataset(path); err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}
	if len(loaded.Dataset) != 1 || loaded.Dataset[0].Id != "Amy" {
		t.Fatalf("Dataset after LoadDataset = %+v, want one entry with Id Amy", loaded.Dataset)
	}

	result, err := loaded.Identify("testdata/amy.jpg")
	if err != nil {
		t.Fatalf("Identify after LoadDataset: %v", err)
	}
	if result.Id != "Amy" {
		t.Errorf("Id = %q, want Amy", result.Id)
	}
}

// TestDownloadModels downloads the real onnxruntime shared library and
// model files and confirms they actually work with Init/RecognizeSingle
// -- not just that files with the right names landed on disk.
func TestDownloadModels(t *testing.T) {

	dir := t.TempDir()

	rec := &onnxface.Recognizer{}
	if err := rec.DownloadModels(dir); err != nil {
		t.Fatalf("DownloadModels: %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir(%s): %v", dir, err)
	}
	if len(entries) == 0 {
		t.Fatalf("DownloadModels populated no files in %s", dir)
	}
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			t.Fatalf("Info(%s): %v", e.Name(), err)
		}
		if info.Size() == 0 {
			t.Errorf("%s downloaded as an empty file", e.Name())
		}
	}

	loaded := &onnxface.Recognizer{}
	if err := loaded.Init(dir); err != nil {
		t.Fatalf("Init with downloaded models: %v", err)
	}
	defer loaded.Close()

	if _, err := loaded.RecognizeSingle("testdata/amy.jpg"); err != nil {
		t.Fatalf("RecognizeSingle with downloaded models: %v", err)
	}

}

// TestDownloadModelsSkipsExisting proves DownloadModels genuinely skips
// files that already exist -- it doesn't just check-then-overwrite. Uses
// a sentinel file instead of a real model, so it needs no network access.
func TestDownloadModelsSkipsExisting(t *testing.T) {

	dir := t.TempDir()

	const sentinelName = "face_detection_yunet_2023mar.onnx"
	sentinel := []byte("not a real model")
	if err := os.WriteFile(filepath.Join(dir, sentinelName), sentinel, 0600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	rec := &onnxface.Recognizer{}
	if err := rec.DownloadModels(dir); err != nil {
		t.Fatalf("DownloadModels: %v", err)
	}

	got, err := os.ReadFile(filepath.Join(dir, sentinelName))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(sentinel) {
		t.Errorf("%s was overwritten, want it left untouched since it already existed", sentinelName)
	}

}
