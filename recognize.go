package onnxface

import (
	"fmt"
	"image"
)

// Result pairs a detected Face with the embedding extracted from it.
type Result struct {
	Face
	Feature []float32
}

/*
Engine ties a FaceDetector and a FaceRecognizer together into a single
convenience API: detect faces, then align and embed each one. Both must
already be constructed (after Init) and are owned by the Engine -- Close
closes both.
*/
type Engine struct {
	Detector   FaceDetector
	Recognizer FaceRecognizer
}

// NewEngine builds an Engine from an already-constructed detector and
// recognizer.
func NewEngine(det FaceDetector, rec FaceRecognizer) *Engine {
	return &Engine{Detector: det, Recognizer: rec}
}

// Close closes both the detector and the recognizer.
func (e *Engine) Close() {

	if e.Detector != nil {
		e.Detector.Close()
	}
	if e.Recognizer != nil {
		e.Recognizer.Close()
	}

}

/*
Recognize detects faces in img and extracts an embedding for each one,
using the Engine's Recognizer to align and embed every detected face.
*/
func (e *Engine) Recognize(img image.Image) ([]Result, error) {

	faces, err := e.Detector.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("detect: %w", err)
	}

	results := make([]Result, len(faces))
	for i, f := range faces {
		aligned := e.Recognizer.Align(img, f.Landmarks)
		feature, err := e.Recognizer.Feature(aligned)
		if err != nil {
			return nil, fmt.Errorf("feature: %w", err)
		}
		results[i] = Result{Face: f, Feature: feature}
	}

	return results, nil

}

// CompareResult is the outcome of comparing two embeddings against a
// tolerance, mirroring go-face/go-recognizer's Tolerance/Distance/
// Confidence vocabulary.
type CompareResult struct {
	// IsMatch is true when Distance is within the given tolerance.
	IsMatch bool
	// Distance is the L2 distance between the (L2-normalized) embeddings;
	// lower means more similar. See Match(..., DistanceL2).
	Distance float64
	// Confidence is a heuristic, not a calibrated probability:
	// 1 - Distance/tolerance, clamped to [0,1]. 1 means an exact match,
	// 0 means at or beyond the tolerance boundary.
	Confidence float64
}

/*
Compare compares two embeddings (as produced by a FaceRecognizer's
Feature) by L2 distance and classifies the result against tolerance --
the caller-chosen accept/reject threshold. OpenCV's own SFace model card
suggests ~1.128 as a starting point for SFace embeddings specifically,
but, as with go-face/go-recognizer's Tolerance, the right value depends on
the deployment (camera quality, lighting) and should be tuned, not
assumed.
*/
func Compare(feature1, feature2 []float32, tolerance float64) CompareResult {

	d := Match(feature1, feature2, DistanceL2)

	return CompareResult{
		IsMatch:    d <= tolerance,
		Distance:   d,
		Confidence: confidenceFor(d, tolerance),
	}

}

// confidenceFor normalizes a distance against the tolerance used to
// accept it, into a convenience [0,1] score where 0 distance is 1.0 and
// a distance at (or past) the tolerance cutoff is 0.0. Not a calibrated
// probability.
func confidenceFor(distance, tolerance float64) float64 {

	if tolerance <= 0 {
		return 0
	}

	c := 1 - distance/tolerance
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}

	return c

}
