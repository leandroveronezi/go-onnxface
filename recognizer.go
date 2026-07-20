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

	"github.com/leandroveronezi/go-onnxface/arcface"
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
	// ErrNoLivenessEngine is returned by CheckLiveness when Init wasn't
	// given a liveness engine (Config.Liveness == LivenessNone).
	ErrNoLivenessEngine = errors.New("no liveness engine configured -- set Config.Liveness before Init")
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
Config selects which engine Recognizer.Init loads for each role
(Detector/Recognizer/Liveness), plus any non-default file names or
per-engine parameters they need. Set fields before calling Init; they
have no effect afterward, since model loading happens inside Init.
*/
type Config struct {
	// Detector selects the detection engine. Empty (DetectorYuNet) is
	// the default.
	Detector DetectorEngine
	// Recognizer selects the recognition engine. Empty (RecognizerSFace)
	// is the default.
	Recognizer RecognizerEngine
	// Liveness selects the liveness engine, if any. Empty (LivenessNone)
	// loads none -- CheckLiveness returns ErrNoLivenessEngine.
	Liveness LivenessEngine

	// DetectorFile overrides the detector's model file name. Empty uses
	// that engine's own default/DownloadModel file name.
	DetectorFile string
	// RecognizerFile overrides the recognizer's model file name. Empty
	// uses that engine's own default/DownloadModel file name --
	// RecognizerArcFace/RecognizerGhostFace ship no weights, so this is
	// required for either of them.
	RecognizerFile string
	// LivenessFiles overrides the liveness engine's two model file names
	// ([0]=primary, [1]=secondary -- both LivenessMiniFAS and
	// LivenessSeetaFace6 need two files). Empty uses that engine's own
	// DownloadModel file names.
	LivenessFiles [2]string

	// ArcFace configures RecognizerArcFace (ignored otherwise) -- see
	// the arcface package doc for how to fill it in from your own model
	// file.
	ArcFace arcface.Config
}

/*
Recognizer is the easy, batteries-included face recognition API: after
Init, work in terms of image file paths -- AddImageToDataset, Identify,
CheckLiveness. For lower-level control (working with image.Image instead
of paths, choosing your own comparison metric), see Engine, Compare, and
the engine subpackages instead -- Recognizer uses them internally.
*/
type Recognizer struct {
	Tolerance float64
	// Model selects the engines and non-default model file names. Set
	// before Init.
	Model Config
	// Dataset holds the known face samples. Mutate it only through
	// AddImageToDataset or LoadDataset -- Identify always matches
	// directly against the current Dataset, so, unlike go-face's
	// classifier, there's no separate "commit" step to remember.
	Dataset []Data

	det  FaceDetector
	rec  FaceRecognizer
	live LivenessDetector
	mu   sync.RWMutex
}

/*
Init loads the engines selected by Model (detector/recognizer, and
liveness if Model.Liveness != LivenessNone) from dir (see DownloadModels)
and, if the ONNX Runtime environment hasn't already been initialized in
this process -- by a previous Recognizer or by direct low-level use --
initializes it automatically using the onnxruntime shared library also
expected in dir.

Set Model before calling Init to choose engines/non-default model file
names. Set Tolerance after calling Init -- Init resets it to a default
(OpenCV's suggested SFace L2 same-person threshold, 1.128; tune it for
your own deployment, same as go-face/go-recognizer's Tolerance).
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

	det, err := newDetectorEngine(r.Model.Detector, dir, r.Model.DetectorFile)
	if err != nil {
		return fmt.Errorf("loading detector model: %w", err)
	}

	rec, err := newRecognizerEngine(r.Model.Recognizer, dir, r.Model.RecognizerFile, r.Model.ArcFace)
	if err != nil {
		det.Close()
		return fmt.Errorf("loading recognizer model: %w", err)
	}

	var live LivenessDetector
	if r.Model.Liveness != LivenessNone {
		live, err = newLivenessEngine(r.Model.Liveness, dir, r.Model.LivenessFiles)
		if err != nil {
			det.Close()
			rec.Close()
			return fmt.Errorf("loading liveness model: %w", err)
		}
	}

	r.det = det
	r.rec = rec
	r.live = live

	return nil

}

/*
Close releases the resources held by the Recognizer's detector,
recognizer, and liveness engine (if any). It does not tear down the
(process-global) ONNX Runtime environment -- call CloseEnvironment
yourself once you're completely done, after closing every
Recognizer/Engine in the process.
*/
func (r *Recognizer) Close() {

	if r.det != nil {
		r.det.Close()
	}
	if r.rec != nil {
		r.rec.Close()
	}
	if r.live != nil {
		r.live.Close()
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

// detectSingleFace finds exactly one face in the image at path, erroring
// if there isn't exactly one -- the shared core of detectSingle (which
// also extracts a recognition embedding) and CheckLiveness (which
// doesn't need one).
func (r *Recognizer) detectSingleFace(path string) (Face, image.Image, error) {

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

	return faces[0], img, nil

}

// detectSingle finds exactly one face in the image at path and its
// embedding, erroring if there isn't exactly one face.
func (r *Recognizer) detectSingle(path string) (Face, []float32, error) {

	f, img, err := r.detectSingleFace(path)
	if err != nil {
		return Face{}, nil, err
	}

	aligned := r.rec.Align(img, f.Landmarks)
	feature, err := r.rec.Feature(aligned)
	if err != nil {
		return Face{}, nil, fmt.Errorf("extracting feature from %s: %w", path, err)
	}

	return f, feature, nil

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

/*
CheckLiveness detects the face in the image at path and classifies it as
a live person or a print/replay spoof, using the engine configured via
Model.Liveness. Returns ErrNoFace/ErrMultipleFaces if the image doesn't
have exactly one face, or ErrNoLivenessEngine if Init wasn't given a
liveness engine (Model.Liveness == LivenessNone) -- check with
errors.Is.
*/
func (r *Recognizer) CheckLiveness(path string) (LivenessResult, error) {

	if r.live == nil {
		return LivenessResult{}, ErrNoLivenessEngine
	}

	f, img, err := r.detectSingleFace(path)
	if err != nil {
		return LivenessResult{}, err
	}

	result, err := r.live.Detect(img, f)
	if err != nil {
		return LivenessResult{}, fmt.Errorf("checking liveness for %s: %w", path, err)
	}

	return result, nil

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
