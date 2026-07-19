package centerface_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/leandroveronezi/go-onnxface/centerface"
)

// TestDownloadModel downloads the real model file into a fresh directory
// and confirms it actually works with NewDetector -- not just that a
// file with the right name landed on disk.
func TestDownloadModel(t *testing.T) {

	initForTest(t)

	dir := t.TempDir()

	if err := centerface.DownloadModel(dir); err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}

	path := filepath.Join(dir, centerface.ModelFileName)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat(%s): %v", path, err)
	}
	if info.Size() == 0 {
		t.Fatalf("%s downloaded as an empty file", path)
	}

	det, err := centerface.NewDetector(path)
	if err != nil {
		t.Fatalf("NewDetector with downloaded model: %v", err)
	}
	defer det.Close()

	img := loadTestImage(t, "../testdata/amy.jpg")
	faces, err := det.Detect(img)
	if err != nil {
		t.Fatalf("Detect with downloaded model: %v", err)
	}
	if len(faces) != 1 {
		t.Fatalf("got %d faces, want 1", len(faces))
	}

}

// TestDownloadModelSkipsExisting proves DownloadModel genuinely skips a
// file that already exists -- it doesn't check-then-overwrite. Uses a
// sentinel file, so it needs no network access.
func TestDownloadModelSkipsExisting(t *testing.T) {

	dir := t.TempDir()

	sentinel := []byte("not a real model")
	path := filepath.Join(dir, centerface.ModelFileName)
	if err := os.WriteFile(path, sentinel, 0600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if err := centerface.DownloadModel(dir); err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(sentinel) {
		t.Errorf("%s was overwritten, want it left untouched since it already existed", path)
	}

}
