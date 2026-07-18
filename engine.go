package onnxface

import "image"

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
