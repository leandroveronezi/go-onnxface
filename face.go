package onnxface

import "github.com/leandroveronezi/go-onnxface/face"

/*
Face, FaceDetector, FaceRecognizer, DistanceType and Match are re-exported
here from the face package (the actual shared contract yunet/sface depend
on) purely so callers of this root package don't need a second import for
them. See the face package doc for the real definitions.
*/
type (
	Face           = face.Face
	FaceDetector   = face.FaceDetector
	FaceRecognizer = face.FaceRecognizer
	DistanceType   = face.DistanceType
)

const (
	DistanceCosine = face.DistanceCosine
	DistanceL2     = face.DistanceL2
)

// Match compares two embeddings produced by a FaceRecognizer. See
// face.Match.
func Match(feature1, feature2 []float32, dist DistanceType) float64 {
	return face.Match(feature1, feature2, dist)
}
