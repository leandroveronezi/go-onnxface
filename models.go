package onnxface

import (
	"os"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface/face"
)

// defaultDetectorModel and defaultRecognizerModel are the file names
// DownloadModels fetches and Recognizer.Init loads by default. Set
// Recognizer.Model before calling Init to use different files instead --
// DownloadModels always fetches these two regardless of Model, the same
// way go-recognizer's DownloadModels always fetches its own hardcoded
// defaults regardless of ModelFiles overrides: a non-default file is
// yours to provide.
const (
	defaultDetectorModel   = "face_detection_yunet_2023mar.onnx"
	defaultRecognizerModel = "face_recognition_sface_2021dec.onnx"
)

// opencvZooRawBase is where the default model files are fetched from --
// the same OpenCV Zoo copies this package was built and validated
// against (see the README for the license of each).
const opencvZooRawBase = "https://github.com/opencv/opencv_zoo/raw/main/models/"

/*
DownloadModels downloads everything Recognizer.Init needs by default into
dir, creating dir if it doesn't exist yet: the onnxruntime shared library
for the current platform, and the default YuNet/SFace model files. For
the other engines (centerface, retinaface), see each package's own
DownloadModel -- kept separate so using one engine never downloads the
others.

Any file that already exists in dir is left untouched and not
re-downloaded, so it's safe to call this on every startup before Init.
*/
func (r *Recognizer) DownloadModels(dir string) error {

	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	if err := downloadRuntimeLibrary(dir); err != nil {
		return err
	}

	for _, name := range []string{defaultDetectorModel, defaultRecognizerModel} {

		path := filepath.Join(dir, name)
		if face.FileExists(path) {
			continue
		}

		if err := downloadModel(path, name); err != nil {
			return err
		}

	}

	return nil

}

// downloadModel fetches the ONNX model file "name" (one of the
// defaultDetectorModel/defaultRecognizerModel constants) from the OpenCV
// Zoo and writes it to path. path is left absent (not a partial file) if
// anything fails partway.
func downloadModel(path, name string) error {

	var modelDir string
	switch name {
	case defaultDetectorModel:
		modelDir = "face_detection_yunet"
	case defaultRecognizerModel:
		modelDir = "face_recognition_sface"
	}

	return face.DownloadFile(path, opencvZooRawBase+modelDir+"/"+name)

}
