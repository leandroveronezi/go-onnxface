package onnxface

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/leandroveronezi/go-onnxface/sface"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

// Sentinel errors for the "expected" failure conditions -- check for
// these with errors.Is instead of matching on the error message text,
// which isn't part of the API contract and may change.
var (
	// ErrNoFace is returned when an image has no detected face, where an
	// operation requires exactly one.
	ErrNoFace = errors.New("no face found")
	// ErrMultipleFaces is returned when an image has more than one
	// detected face, where an operation requires exactly one.
	ErrMultipleFaces = errors.New("more than one face found")
	// ErrNoMatch is returned by Identify when the detected face doesn't
	// match any Dataset entry within Tolerance.
	ErrNoMatch = errors.New("no match within tolerance")
)

// Data is one known face in a Recognizer's Dataset.
type Data struct {
	Id      string
	Feature []float32
}

// Recognition is a detected face matched against a Recognizer's Dataset.
type Recognition struct {
	Data
	Rectangle  image.Rectangle
	Landmarks  [5]image.Point
	Distance   float64
	Confidence float64
}

/*
ModelFiles selects non-default model file names, both resolved relative
to the dir passed to Init. Set fields before calling Init; they have no
effect afterward, since model loading happens inside Init. DownloadModels
ignores these and always fetches the defaults -- a non-default file is
yours to provide.
*/
type ModelFiles struct {
	// Detector is the YuNet detection model file name. Empty (the
	// default) uses face_detection_yunet_2023mar.onnx.
	Detector string
	// Recognizer is the SFace recognition model file name. Empty (the
	// default) uses face_recognition_sface_2021dec.onnx.
	Recognizer string
}

/*
Recognizer is the easy, batteries-included face recognition API: after
Init, work in terms of image file paths -- AddImageToDataset, Identify.
For lower-level control (custom detectors/recognizers, working with
image.Image instead of paths, choosing your own comparison metric), see
Engine, Compare, and the yunet/sface packages instead -- Recognizer uses
them internally.
*/
type Recognizer struct {
	Tolerance float64
	// Model selects non-default model file names. Set before Init.
	Model ModelFiles
	// Dataset holds the known face samples. Mutate it only through
	// AddImageToDataset or LoadDataset -- Identify always matches
	// directly against the current Dataset, so, unlike go-face's
	// classifier, there's no separate "commit" step to remember.
	Dataset []Data

	det FaceDetector
	rec FaceRecognizer
	mu  sync.RWMutex
}

/*
Init loads the detector/recognizer models from dir (see DownloadModels)
and, if the ONNX Runtime environment hasn't already been initialized in
this process -- by a previous Recognizer or by direct low-level use --
initializes it automatically using the onnxruntime shared library also
expected in dir.

Set Model before calling Init to choose non-default model file names. Set
Tolerance after calling Init -- Init resets it to a default (OpenCV's
suggested SFace L2 same-person threshold, 1.128; tune it for your own
deployment, same as go-face/go-recognizer's Tolerance).
*/
func (r *Recognizer) Init(dir string) error {

	r.Tolerance = 1.128

	r.mu.Lock()
	r.Dataset = make([]Data, 0)
	r.mu.Unlock()

	if !IsInitialized() {
		libName, err := runtimeLibraryName()
		if err != nil {
			return err
		}
		if err := InitEnvironment(filepath.Join(dir, libName)); err != nil {
			return fmt.Errorf("initializing onnx runtime: %w", err)
		}
	}

	detectorModel := r.Model.Detector
	if detectorModel == "" {
		detectorModel = defaultDetectorModel
	}
	recognizerModel := r.Model.Recognizer
	if recognizerModel == "" {
		recognizerModel = defaultRecognizerModel
	}

	det, err := yunet.NewDetector(filepath.Join(dir, detectorModel))
	if err != nil {
		return fmt.Errorf("loading detector model: %w", err)
	}

	rec, err := sface.NewRecognizer(filepath.Join(dir, recognizerModel))
	if err != nil {
		det.Close()
		return fmt.Errorf("loading recognizer model: %w", err)
	}

	r.det = det
	r.rec = rec

	return nil

}

/*
Close releases the resources held by the Recognizer's detector and
recognizer. It does not tear down the (process-global) ONNX Runtime
environment -- call CloseEnvironment yourself once you're completely
done, after closing every Recognizer/Engine in the process.
*/
func (r *Recognizer) Close() {

	if r.det != nil {
		r.det.Close()
	}
	if r.rec != nil {
		r.rec.Close()
	}

}

// loadImage decodes the image file at path (JPEG or PNG).
func loadImage(path string) (image.Image, error) {

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	return img, err

}

// detectSingle finds exactly one face in the image at path and its
// embedding, erroring if there isn't exactly one face.
func (r *Recognizer) detectSingle(path string) (Face, []float32, error) {

	img, err := loadImage(path)
	if err != nil {
		return Face{}, nil, fmt.Errorf("loading %s: %w", path, err)
	}

	faces, err := r.det.Detect(img)
	if err != nil {
		return Face{}, nil, fmt.Errorf("detecting faces in %s: %w", path, err)
	}
	if len(faces) == 0 {
		return Face{}, nil, fmt.Errorf("%w in %s", ErrNoFace, path)
	}
	if len(faces) > 1 {
		return Face{}, nil, fmt.Errorf("%w in %s", ErrMultipleFaces, path)
	}

	aligned := r.rec.Align(img, faces[0].Landmarks)
	feature, err := r.rec.Feature(aligned)
	if err != nil {
		return Face{}, nil, fmt.Errorf("extracting feature from %s: %w", path, err)
	}

	return faces[0], feature, nil

}

// AddImageToDataset adds a sample image to the Dataset. The image must
// contain exactly one face -- returns ErrNoFace or ErrMultipleFaces
// otherwise (check with errors.Is).
func (r *Recognizer) AddImageToDataset(path, id string) error {

	_, feature, err := r.detectSingle(path)
	if err != nil {
		return err
	}

	r.mu.Lock()
	r.Dataset = append(r.Dataset, Data{Id: id, Feature: feature})
	r.mu.Unlock()

	return nil

}

/*
RecognizeSingle detects the face in the image at path, returning
ErrNoFace or ErrMultipleFaces unless there's exactly one (check with
errors.Is).
*/
func (r *Recognizer) RecognizeSingle(path string) (Face, error) {

	f, _, err := r.detectSingle(path)
	return f, err

}

/*
RecognizeMultiples returns every face detected in the image at path, with
no identity matching against Dataset -- see Identify for that.
*/
func (r *Recognizer) RecognizeMultiples(path string) ([]Face, error) {

	img, err := loadImage(path)
	if err != nil {
		return nil, fmt.Errorf("loading %s: %w", path, err)
	}

	faces, err := r.det.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("detecting faces in %s: %w", path, err)
	}

	return faces, nil

}

/*
Identify detects the face in the image at path and matches it against
Dataset. Returns ErrNoFace/ErrMultipleFaces if there isn't exactly one
face, or ErrNoMatch if no Dataset entry is within Tolerance -- check
with errors.Is.
*/
func (r *Recognizer) Identify(path string) (Recognition, error) {

	f, feature, err := r.detectSingle(path)
	if err != nil {
		return Recognition{}, err
	}

	data, distance, ok := r.bestMatch(feature)
	if !ok {
		return Recognition{}, fmt.Errorf("%w for %s", ErrNoMatch, path)
	}

	return Recognition{
		Data:       data,
		Rectangle:  f.Rectangle,
		Landmarks:  f.Landmarks,
		Distance:   distance,
		Confidence: confidenceFor(distance, r.Tolerance),
	}, nil

}

/*
IdentifyMultiples detects every face in the image at path and matches
each against Dataset, skipping faces with no Dataset entry within
Tolerance. An empty slice (not an error) is returned if no face matched.
*/
func (r *Recognizer) IdentifyMultiples(path string) ([]Recognition, error) {

	faces, err := r.RecognizeMultiples(path)
	if err != nil {
		return nil, err
	}

	img, err := loadImage(path)
	if err != nil {
		return nil, fmt.Errorf("loading %s: %w", path, err)
	}

	recognitions := make([]Recognition, 0, len(faces))
	for _, f := range faces {

		aligned := r.rec.Align(img, f.Landmarks)
		feature, err := r.rec.Feature(aligned)
		if err != nil {
			return nil, fmt.Errorf("extracting feature from %s: %w", path, err)
		}

		data, distance, ok := r.bestMatch(feature)
		if !ok {
			continue
		}

		recognitions = append(recognitions, Recognition{
			Data:       data,
			Rectangle:  f.Rectangle,
			Landmarks:  f.Landmarks,
			Distance:   distance,
			Confidence: confidenceFor(distance, r.Tolerance),
		})

	}

	return recognitions, nil

}

// bestMatch finds the closest Dataset entry to feature by L2 distance,
// reporting ok=false if Dataset is empty or the closest entry is beyond
// Tolerance.
func (r *Recognizer) bestMatch(feature []float32) (data Data, distance float64, ok bool) {

	r.mu.RLock()
	defer r.mu.RUnlock()

	bestDist := math.Inf(1)
	bestIdx := -1
	for i, d := range r.Dataset {
		dist := Match(feature, d.Feature, DistanceL2)
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}

	if bestIdx < 0 || bestDist > r.Tolerance {
		return Data{}, bestDist, false
	}

	return r.Dataset[bestIdx], bestDist, true

}
