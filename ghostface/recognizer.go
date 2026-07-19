/*
Package ghostface implements face recognition using GhostFaceNetV1
(GhostFaceNets, MIT licensed -- https://github.com/HamadYA/GhostFaceNets),
a lightweight, modern (2023) architecture trained with ArcFace loss on
MS1MV2/MS1MV3, competitive with ArcFace itself (~99.7% LFW per the
authors).

Same licensing situation as arcface, and the same answer: the wrapper
code here is MIT, but the published weights
(GhostFaceNet_W1.3_S1_ArcFace.h5) are trained on MS1MV2/MS1MV3 -- the
same research-only InsightFace-lineage data ArcFace/Buffalo_L carry.
This package doesn't ship, host, or download that weight file. Bring
your own converted .onnx, and see the main README's Licensing section
before using it commercially.

To reproduce the ONNX file this package expects ("input"/"embedding",
NHWC 112x112x3 float32 in, 512-d out) from the original Keras weights:
download GhostFaceNet_W1.3_S1_ArcFace.h5 from
https://github.com/HamadYA/GhostFaceNets/releases/tag/v1.2, rebuild
GhostFaceNetV1's architecture from
https://github.com/serengil/deepface/blob/master/deepface/models/facial_recognition/GhostFaceNet.py
(load_weights, not load_model -- the .h5 predates current Keras'
config-deserialization format), then export with tf2onnx:

	model.load_weights("GhostFaceNet_W1.3_S1_ArcFace.h5")
	spec = (tf.TensorSpec((1, 112, 112, 3), tf.float32, name="input"),)
	tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path="ghostfacenet.onnx")

Recognizer implements the face.FaceRecognizer contract.
*/
package ghostface

import (
	"fmt"
	"image"

	"github.com/leandroveronezi/go-onnxface/face"
	ort "github.com/yalue/onnxruntime_go"
)

var _ face.FaceRecognizer = (*Recognizer)(nil)

const alignedSize = face.AlignedSize

// featureSize is the dimensionality of the GhostFaceNet embedding.
const featureSize = 512

// Recognizer runs GhostFaceNetV1, producing a 512-d embedding for an
// aligned face crop.
type Recognizer struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

// NewRecognizer loads the GhostFaceNetV1 ONNX model from modelPath (see
// the package doc for where to get and how to convert the weights).
// InitEnvironment must have been called first.
func NewRecognizer(modelPath string) (*Recognizer, error) {

	r := &Recognizer{}

	var err error
	// NHWC, unlike every other engine in this repo (which are all
	// PyTorch/OpenCV-derived NCHW) -- GhostFaceNetV1 is a TensorFlow/
	// Keras model, exported via tf2onnx, which preserves TF's native
	// channel-last layout.
	r.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, alignedSize, alignedSize, 3))
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
		[]string{"input"},
		[]string{"embedding"},
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

/*
Align implements face.FaceRecognizer via face.AlignCrop -- the same
112x112 5-point template ArcFace/SFace use. GhostFaceNets' own training
data (faces_emore_112x112_folders) was prepared with InsightFace's
alignment toolchain, which uses this exact template -- not the simpler
eye-rotation alignment DeepFace's own generic pipeline applies by
default, which is a worse match for what the model actually saw during
training.
*/
func (r *Recognizer) Align(img image.Image, landmarks [5]image.Point) image.Image {
	return face.AlignCrop(img, landmarks)
}

/*
Feature extracts a 512-d embedding from an aligned 112x112 face crop (as
produced by Align/face.AlignCrop). InitEnvironment must have been called
first.
*/
func (r *Recognizer) Feature(aligned image.Image) ([]float32, error) {

	b := aligned.Bounds()
	if b.Dx() != alignedSize || b.Dy() != alignedSize {
		return nil, fmt.Errorf("aligned image must be %dx%d, got %dx%d", alignedSize, alignedSize, b.Dx(), b.Dy())
	}

	data := r.input.GetData()
	for y := 0; y < alignedSize; y++ {
		for x := 0; x < alignedSize; x++ {
			rr, gg, bb, _ := aligned.At(b.Min.X+x, b.Min.Y+y).RGBA()
			idx := (y*alignedSize + x) * 3
			// DeepFace's extract_faces returns RGB; GhostFaceNet has no
			// explicit normalization entry in DeepFace's own
			// normalize_input (falls through to "base"), which -- given
			// extract_faces already scales to [0,1] -- means a plain
			// /255, no mean subtraction.
			data[idx+0] = float32(rr>>8) / 255
			data[idx+1] = float32(gg>>8) / 255
			data[idx+2] = float32(bb>>8) / 255
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
