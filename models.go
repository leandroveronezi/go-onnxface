package onnxface

import (
	"os"
	"path/filepath"

	"github.com/leandroveronezi/go-onnxface/centerface"
	"github.com/leandroveronezi/go-onnxface/face"
	"github.com/leandroveronezi/go-onnxface/liveness"
	"github.com/leandroveronezi/go-onnxface/retinaface"
	"github.com/leandroveronezi/go-onnxface/seetaface6"
)

// defaultDetectorModel and defaultRecognizerModel are the file names
// DownloadModels fetches and Recognizer.Init loads for DetectorYuNet/
// RecognizerSFace (the defaults). Set Recognizer.Model.DetectorFile/
// RecognizerFile before calling Init to use different files for those
// same two engines instead -- a non-default file is yours to provide.
const (
	defaultDetectorModel   = "face_detection_yunet_2023mar.onnx"
	defaultRecognizerModel = "face_recognition_sface_2021dec.onnx"
)

// opencvZooRawBase is where the default model files are fetched from --
// the same OpenCV Zoo copies this package was built and validated
// against (see the README for the license of each).
const opencvZooRawBase = "https://github.com/opencv/opencv_zoo/raw/main/models/"

/*
DownloadModels downloads everything Recognizer.Init needs into dir,
creating dir if it doesn't exist yet: the onnxruntime shared library for
the current platform, plus whatever Model.Detector/Recognizer/Liveness
selects (defaulting to YuNet+SFace, no liveness engine).
RecognizerArcFace/RecognizerGhostFace ship no weights -- DownloadModels
is a no-op for them, and Init will error if Model.RecognizerFile isn't
set.

Set Model before calling DownloadModels so it fetches the same engines
Init will load. Any file that already exists in dir is left untouched
and not re-downloaded, so it's safe to call this on every startup before
Init.
*/
func (r *Recognizer) DownloadModels(dir string) error {

	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	if err := downloadRuntimeLibrary(dir); err != nil {
		return err
	}

	switch r.Model.Detector {
	case DetectorCenterFace:
		if err := centerface.DownloadModel(dir); err != nil {
			return err
		}
	case DetectorRetinaFace:
		if err := retinaface.DownloadModel(dir); err != nil {
			return err
		}
	default:
		if err := downloadDefaultModel(dir, defaultDetectorModel); err != nil {
			return err
		}
	}

	switch r.Model.Recognizer {
	case RecognizerArcFace, RecognizerGhostFace:
		// no downloadable weights -- bring your own file.
	default:
		if err := downloadDefaultModel(dir, defaultRecognizerModel); err != nil {
			return err
		}
	}

	switch r.Model.Liveness {
	case LivenessMiniFAS:
		if err := liveness.DownloadModel(dir); err != nil {
			return err
		}
	case LivenessSeetaFace6:
		if err := seetaface6.DownloadModel(dir); err != nil {
			return err
		}
	}

	return nil

}

func downloadDefaultModel(dir, name string) error {

	path := filepath.Join(dir, name)
	if face.FileExists(path) {
		return nil
	}

	return downloadModel(path, name)

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
