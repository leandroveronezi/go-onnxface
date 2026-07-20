package seetaface6

import (
	"os"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface/face"
)

// FasFirstFileName and FasSecondFileName are the file names DownloadModel
// writes to and NewDetector expects by convention (though NewDetector
// takes any paths, so these are just what DownloadModel happens to use).
const (
	FasFirstFileName  = "fas_first.onnx"
	FasSecondFileName = "fas_second.onnx"
)

// modelBaseURL points at go-onnxface's own release -- SeetaFace6 only
// publishes the proprietary .csta format, there's no upstream ONNX
// download.
const modelBaseURL = "https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/"

/*
DownloadModel downloads both models into dir (creating it if needed),
skipping any file that's already there. Both are needed together --
Detect always runs fas_second first and may call fas_first depending on
the result -- so unlike the other engines' DownloadModel, there's no
"only fetch what you use" split to make here.
*/
func DownloadModel(dir string) error {

	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	for _, name := range []string{FasFirstFileName, FasSecondFileName} {

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
