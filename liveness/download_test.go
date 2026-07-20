package liveness_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/leandroveronezi/go-onnxface/liveness"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

// TestDownloadModel downloads the real model files into a fresh
// directory and confirms they actually work with NewDetector -- not
// just that files with the right names landed on disk.
func TestDownloadModel(t *testing.T) {

	initForTest(t)

	dir := t.TempDir()

	if err := liveness.DownloadModel(dir); err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}

	for _, name := range []string{liveness.ModelV2FileName, liveness.ModelV1SEFileName} {
		path := filepath.Join(dir, name)
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("Stat(%s): %v", path, err)
		}
		if info.Size() == 0 {
			t.Fatalf("%s downloaded as an empty file", path)
		}
	}

	yn, err := yunet.NewDetector(yunetModelPath)
	if err != nil {
		t.Fatalf("yunet.NewDetector: %v", err)
	}
	defer yn.Close()

	img := loadTestImage(t, "../testdata/amy.jpg")
	faces, err := yn.Detect(img)
	if err != nil || len(faces) != 1 {
		t.Fatalf("Detect: %d faces, err=%v", len(faces), err)
	}

	det, err := liveness.NewDetector(
		filepath.Join(dir, liveness.ModelV2FileName),
		filepath.Join(dir, liveness.ModelV1SEFileName),
	)
	if err != nil {
		t.Fatalf("NewDetector with downloaded models: %v", err)
	}
	defer det.Close()

	if _, err := det.Detect(img, faces[0]); err != nil {
		t.Fatalf("Detect with downloaded models: %v", err)
	}

}

// TestDownloadModelSkipsExisting proves DownloadModel genuinely skips
// files that already exist -- it doesn't check-then-overwrite. Uses
// sentinel files, so it needs no network access.
func TestDownloadModelSkipsExisting(t *testing.T) {

	dir := t.TempDir()

	sentinels := map[string][]byte{
		liveness.ModelV2FileName:   []byte("not a real model"),
		liveness.ModelV1SEFileName: []byte("also not a real model"),
	}
	for name, data := range sentinels {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatalf("WriteFile(%s): %v", name, err)
		}
	}

	if err := liveness.DownloadModel(dir); err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}

	for name, want := range sentinels {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", name, err)
		}
		if string(got) != string(want) {
			t.Errorf("%s was overwritten, want it left untouched since it already existed", name)
		}
	}

}
