// Package sface implements face recognition using SFace
// (face_recognition_sface_2021dec.onnx from the OpenCV Zoo, Apache-2.0
// licensed). Recognizer implements the face.FaceRecognizer contract.
package sface

import (
	"fmt"
	"image"

	"github.com/leandroveronezi/go-onnxface/face"
	ort "github.com/yalue/onnxruntime_go"
)

var _ face.FaceRecognizer = (*Recognizer)(nil)

// alignedSize is the fixed crop size cv::FaceRecognizerSF::alignCrop warps
// faces to before feeding them to the SFace network.
const alignedSize = face.AlignedSize

// featureSize is the dimensionality of the SFace embedding (the "fc1"
// output of face_recognition_sface_2021dec.onnx).
const featureSize = 128

// Recognizer runs SFace face recognition, producing a 128-d embedding for
// an aligned face crop.
type Recognizer struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

/*
NewRecognizer loads the SFace face recognition model from modelPath (e.g.
face_recognition_sface_2021dec.onnx from the OpenCV Zoo). InitEnvironment
must have been called first.
*/
func NewRecognizer(modelPath string) (*Recognizer, error) {

	r := &Recognizer{}

	var err error
	r.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, alignedSize, alignedSize))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}
	r.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, featureSize))
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("allocating output tensor: %w", err)
	}

	r.session, err = ort.NewAdvancedSession(
		modelPath,
		[]string{"data"},
		[]string{"fc1"},
		[]ort.Value{r.input},
		[]ort.Value{r.output},
		nil,
	)
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return r, nil

}

// Close releases the resources held by the recognizer.
func (r *Recognizer) Close() {

	if r.session != nil {
		r.session.Destroy()
	}
	if r.input != nil {
		r.input.Destroy()
	}
	if r.output != nil {
		r.output.Destroy()
	}

}

// Align implements face.FaceRecognizer via face.AlignCrop -- SFace uses
// the same 112x112 5-point template as the rest of the package family.
func (r *Recognizer) Align(img image.Image, landmarks [5]image.Point) image.Image {
	return face.AlignCrop(img, landmarks)
}

/*
Feature extracts a 128-d embedding from an aligned 112x112 face crop (as
produced by Align/face.AlignCrop). InitEnvironment must have been called
first.
*/
func (r *Recognizer) Feature(aligned image.Image) ([]float32, error) {

	b := aligned.Bounds()
	if b.Dx() != alignedSize || b.Dy() != alignedSize {
		return nil, fmt.Errorf("aligned image must be %dx%d, got %dx%d", alignedSize, alignedSize, b.Dx(), b.Dy())
	}

	data := r.input.GetData()
	frameSize := alignedSize * alignedSize
	for y := 0; y < alignedSize; y++ {
		for x := 0; x < alignedSize; x++ {
			rr, gg, bb, _ := aligned.At(b.Min.X+x, b.Min.Y+y).RGBA()
			idx := y*alignedSize + x
			// blobFromImage(_aligned_img, 1, Size(112,112), Scalar(0,0,0),
			// swapRB=true, crop=false): scalefactor=1 means no /255
			// normalization; swapRB=true on an OpenCV (BGR) source yields
			// an RGB-ordered blob, which is exactly the channel order
			// image.Image.At().RGBA() already gives us.
			data[0*frameSize+idx] = float32(rr >> 8)
			data[1*frameSize+idx] = float32(gg >> 8)
			data[2*frameSize+idx] = float32(bb >> 8)
		}
	}

	if err := r.session.Run(); err != nil {
		return nil, fmt.Errorf("running session: %w", err)
	}

	out := r.output.GetData()
	feature := make([]float32, len(out))
	copy(feature, out)
	return feature, nil

}
