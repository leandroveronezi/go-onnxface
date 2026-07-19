/*
Package arcface is a bridge for ArcFace-family recognition models (e.g.
InsightFace's buffalo_l/antelopev2), in the standard ONNX export
convention -- not a specific model. It ships no weights and does not
download any: bring your own .onnx file.

Read this before pointing it at a real model: almost every published
ArcFace-family weight file (including InsightFace's own buffalo_l and
antelopev2) is trained on MS1M/CASIA-WebFace-lineage datasets released
"for research purposes only" -- that restriction applies to the trained
weights regardless of this wrapper code's MIT license, or of what license
the specific .onnx file's own repository claims. A commercial deployment
needs a commercial license for whatever weights you load (InsightFace
sells one for buffalo_l) -- this package doesn't change that, it just
knows how to run the ONNX format. See the README's Licensing section for
the full picture, and prefer the sface package for anything you don't
have explicit commercial rights to.

Recognizer implements the face.FaceRecognizer contract.
*/
package arcface

import (
	"fmt"
	"image"

	"github.com/leandroveronezi/go-onnxface/face"
	ort "github.com/yalue/onnxruntime_go"
)

var _ face.FaceRecognizer = (*Recognizer)(nil)

const alignedSize = face.AlignedSize

/*
Config configures a Recognizer for the specific ONNX export it's pointed
at. Unlike yunet/sface/centerface (which each ship one specific model
file this package already knows the tensor names for), ArcFace exports
vary -- inspect your own model file to fill this in, e.g.:

	python3 -c "import onnxruntime as ort; s = ort.InferenceSession('model.onnx'); print(s.get_inputs()[0].name, s.get_outputs()[0].name)"
*/
type Config struct {
	// InputName is the ONNX graph's input tensor name (e.g. "data" or
	// "input.1"). Required.
	InputName string
	// OutputName is the ONNX graph's embedding output tensor name (e.g.
	// "fc1", or an exporter-assigned name like "683"). Required.
	OutputName string
	// InputMean and InputStd normalize pixels to
	// (pixel-InputMean)/InputStd before inference. Both default to 127.5
	// (the standard convention for ONNX-exported ArcFace models) when
	// left zero. MXNet-originated exports sometimes use 0/1 instead --
	// see InsightFace's own ArcFaceONNX class for the heuristic (looking
	// for Sub/Mul nodes near the graph input) it uses to pick between
	// the two; this package doesn't replicate that heuristic, so set it
	// explicitly if your export needs it.
	InputMean float64
	InputStd  float64
}

// Recognizer runs an ArcFace-family recognition model, producing an
// embedding of whatever dimensionality that model was trained to
// produce (512-d is the most common).
type Recognizer struct {
	session    *ort.DynamicAdvancedSession
	inputName  string
	outputName string
	mean, std  float32
}

// NewRecognizer loads modelPath (see Config for how to fill in the
// required fields). InitEnvironment must have been called first.
func NewRecognizer(modelPath string, cfg Config) (*Recognizer, error) {

	if cfg.InputName == "" {
		return nil, fmt.Errorf("arcface: Config.InputName is required -- see the package doc for how to find it")
	}
	if cfg.OutputName == "" {
		return nil, fmt.Errorf("arcface: Config.OutputName is required -- see the package doc for how to find it")
	}

	mean, std := cfg.InputMean, cfg.InputStd
	if mean == 0 {
		mean = 127.5
	}
	if std == 0 {
		std = 127.5
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{cfg.InputName},
		[]string{cfg.OutputName},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return &Recognizer{
		session:    session,
		inputName:  cfg.InputName,
		outputName: cfg.OutputName,
		mean:       float32(mean),
		std:        float32(std),
	}, nil

}

// Close releases the resources held by the recognizer.
func (r *Recognizer) Close() {
	if r.session != nil {
		r.session.Destroy()
	}
}

// Align implements face.FaceRecognizer via face.AlignCrop -- the
// ArcFace family uses the same 112x112 5-point template as SFace
// (confirmed identical to InsightFace's own face_align.arcface_dst).
func (r *Recognizer) Align(img image.Image, landmarks [5]image.Point) image.Image {
	return face.AlignCrop(img, landmarks)
}

/*
Feature extracts an embedding from an aligned 112x112 face crop (as
produced by Align/face.AlignCrop). InitEnvironment must have been called
first.
*/
func (r *Recognizer) Feature(aligned image.Image) ([]float32, error) {

	b := aligned.Bounds()
	if b.Dx() != alignedSize || b.Dy() != alignedSize {
		return nil, fmt.Errorf("aligned image must be %dx%d, got %dx%d", alignedSize, alignedSize, b.Dx(), b.Dy())
	}

	input, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, alignedSize, alignedSize))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}
	defer input.Destroy()

	data := input.GetData()
	frameSize := alignedSize * alignedSize
	for y := 0; y < alignedSize; y++ {
		for x := 0; x < alignedSize; x++ {
			rr, gg, bb, _ := aligned.At(b.Min.X+x, b.Min.Y+y).RGBA()
			idx := y*alignedSize + x
			// InsightFace's ArcFaceONNX.get_feat: blobFromImage(img,
			// 1/std, size, (mean,mean,mean), swapRB=True) -- RGB order
			// (matches image.Image.At().RGBA() already), then
			// (pixel-mean)/std.
			data[0*frameSize+idx] = (float32(rr>>8) - r.mean) / r.std
			data[1*frameSize+idx] = (float32(gg>>8) - r.mean) / r.std
			data[2*frameSize+idx] = (float32(bb>>8) - r.mean) / r.std
		}
	}

	outputs := []ort.Value{nil}
	if err := r.session.Run([]ort.Value{input}, outputs); err != nil {
		return nil, fmt.Errorf("running session: %w", err)
	}
	defer outputs[0].Destroy()

	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for output %q", r.outputName)
	}

	raw := out.GetData()
	feature := make([]float32, len(raw))
	copy(feature, raw)
	return feature, nil

}
