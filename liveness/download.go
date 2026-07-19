package liveness

import (
	"os"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface/face"
)

// ModelV2FileName and ModelV1SEFileName are the file names DownloadModel
// writes to and NewDetector expects by convention (though NewDetector
// takes any paths, so these are just what DownloadModel happens to use).
const (
	ModelV2FileName   = "minifasnet_v2.onnx"
	ModelV1SEFileName = "minifasnet_v1se.onnx"
)

// modelBaseURL points at go-onnxface's own release -- minivision-ai only
// publish PyTorch .pth weights, there's no upstream ONNX download. See
// NewDetector's doc for how these were produced.
const modelBaseURL = "https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/"

/*
DownloadModel downloads both ensemble models into dir (creating it if
needed), skipping any file that's already there. Both are needed
together -- Detect always runs the full two-model ensemble -- so unlike
the other engines' DownloadModel, there's no "only fetch what you use"
split to make here.
*/
func DownloadModel(dir string) error {

	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	for _, name := range []string{ModelV2FileName, ModelV1SEFileName} {

		path := filepath.Join(dir, name)
		if face.FileExists(path) {
			continue
		}

		if err := face.DownloadFile(path, modelBaseURL+name); err != nil {
			return err
		}

	}

	return nil

}
