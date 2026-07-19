package centerface

import (
	"os"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface/face"
)

// ModelFileName is the file name DownloadModel writes to and NewDetector
// expects by convention (though NewDetector takes any path, so this is
// just what DownloadModel happens to use).
const ModelFileName = "centerface.onnx"

// modelURL points at go-onnxface's own release, not Star-Clouds' -- the
// published centerface.onnx has a fixed [10,3,32,32] input (see
// NewDetector's doc); this asset is that same file with only its shape
// metadata relaxed to dynamic dimensions, weights unmodified.
const modelURL = "https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/centerface.onnx"

/*
DownloadModel downloads centerface.onnx into dir (creating it if needed),
skipping the download if it's already there. Kept separate from
onnxface.Recognizer.DownloadModels -- a caller using centerface instead
of yunet never needs to fetch yunet/sface too, and vice versa.
*/
func DownloadModel(dir string) error {

	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	path := filepath.Join(dir, ModelFileName)
	if face.FileExists(path) {
		return nil
	}

	return face.DownloadFile(path, modelURL)

}
