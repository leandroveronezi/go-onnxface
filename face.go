package onnxface

import (
	"image"
	"math"
)

// Face is a detected face, in the coordinate space of the image passed to
// a FaceDetector's Detect.
type Face struct {
	Rectangle image.Rectangle
	// Landmarks holds 5 points, in order: right eye, left eye, nose tip,
	// right corner of mouth, left corner of mouth -- the standard 5-point
	// layout shared by YuNet, SCRFD, RetinaFace and the ArcFace family
	// ("right"/"left" from the subject's own perspective, so "right eye"
	// appears on the left side of a front-facing photo).
	Landmarks [5]image.Point
	Score     float32
}

// DistanceType selects the metric Match uses to compare two embeddings.
type DistanceType int

const (
	// DistanceCosine is cosine similarity: 1 for identical direction, 0
	// for orthogonal, higher means more similar.
	DistanceCosine DistanceType = iota
	// DistanceL2 is Euclidean distance between L2-normalized embeddings:
	// 0 for identical direction, lower means more similar.
	DistanceL2
)

/*
Match compares two embeddings produced by a FaceRecognizer (they must be
the same length -- i.e. come from the same recognizer). Both are
L2-normalized first, then compared by cosine similarity or L2 distance.
This is a generic vector comparison, independent of which recognizer
produced the embeddings.
*/
func Match(feature1, feature2 []float32, dist DistanceType) float64 {

	n1 := normalize(feature1)
	n2 := normalize(feature2)

	switch dist {
	case DistanceL2:
		var sum float64
		for i := range n1 {
			d := n1[i] - n2[i]
			sum += d * d
		}
		return math.Sqrt(sum)
	default:
		var sum float64
		for i := range n1 {
			sum += n1[i] * n2[i]
		}
		return sum
	}

}

func normalize(f []float32) []float64 {

	out := make([]float64, len(f))
	var sumSq float64
	for i, v := range f {
		out[i] = float64(v)
		sumSq += out[i] * out[i]
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 {
		return out
	}
	for i := range out {
		out[i] /= norm
	}
	return out

}
