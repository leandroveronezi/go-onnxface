// Package face holds the contract shared by every detection/recognition
// engine in go-onnxface (yunet, sface, and future additions): the Face
// type, the FaceDetector/FaceRecognizer interfaces, and generic embedding
// comparison (Match). It has no dependencies on any specific model, so
// engine packages can depend on it without creating an import cycle back
// to the root package, which depends on the engines.
package face

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

/*
FaceDetector locates faces in an image. yunet.Detector implements this.
The interface exists so a future detector (a different ONNX model, with
its own pre/post-processing) can be swapped in without changing code that
only depends on the contract.
*/
type FaceDetector interface {
	Detect(img image.Image) ([]Face, error)
	Close()
}

/*
FaceRecognizer prepares a detected face and extracts a fixed-length
embedding from it. sface.Recognizer implements this. The interface exists
so a future recognizer -- e.g. an ArcFace-family model, should a
commercially-usable license become available -- can be swapped in: Align
is part of the contract (not a free-standing function a caller must
remember to pair with the right recognizer) because different recognizers
can expect different crops/templates; the embedding length is
implementation-defined ([]float32 of whatever dimensionality that model
produces), and Match already operates on plain []float32, not a fixed
size.
*/
type FaceRecognizer interface {
	// Align crops/warps img into whatever input the recognizer expects,
	// using the landmarks a FaceDetector produced.
	Align(img image.Image, landmarks [5]image.Point) image.Image
	Feature(aligned image.Image) ([]float32, error)
	Close()
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
