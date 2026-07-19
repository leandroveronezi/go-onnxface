package retinaface

import (
	"os"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface/face"
)

// ModelFileName is the file name DownloadModel writes to and NewDetector
// expects by convention (though NewDetector takes any path, so this is
// just what DownloadModel happens to use).
const ModelFileName = "retinaface.onnx"

// modelURL points at go-onnxface's own release -- InsightFace/biubug6
// only publish PyTorch .pth weights, there's no upstream ONNX file to
// point at directly. See NewDetector's doc for how this one was produced.
const modelURL = "https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/retinaface.onnx"

/*
DownloadModel downloads retinaface.onnx into dir (creating it if
needed), skipping the download if it's already there. Kept separate from
onnxface.Recognizer.DownloadModels -- a caller using retinaface instead
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
