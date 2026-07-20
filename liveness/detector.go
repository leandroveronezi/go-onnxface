/*
Package liveness implements print/replay spoof detection using the
MiniFASNetV2/MiniFASNetV1SE ensemble from minivision-ai's
Silent-Face-Anti-Spoofing (Apache-2.0, trained for this exact task --
not a face-identity dataset, so none of the licensing concerns that
apply to recognition models apply here).

Unlike the face.FaceDetector/FaceRecognizer engines, Detector isn't tied
to that contract: it takes a face.Face from any detector (or built by
hand) and classifies whether that region is a live face or a
print/screen spoof, independent of who found it. It implements
face.LivenessDetector, the same contract seetaface6.Detector does.
*/
package liveness

import (
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"

	"github.com/leandroveronezi/go-onnxface/face"
	ort "github.com/yalue/onnxruntime_go"
)

// cropSize is the fixed input resolution both ensemble models expect.
const cropSize = 80

// model pairs one ONNX session with the crop scale
// Silent-Face-Anti-Spoofing's own model filename encodes for it (e.g.
// "2.7_80x80_MiniFASNetV2.pth" -> scale 2.7): how far out from the
// detected face box this particular model wants its input cropped from.
type model struct {
	scale   float64
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

// Result is the outcome of a liveness check -- an alias for face.Result,
// the contract shared with seetaface6.Detector (see face.LivenessDetector).
type Result = face.Result

// Detector runs the MiniFASNetV2/MiniFASNetV1SE ensemble. Implements
// face.LivenessDetector.
type Detector struct {
	models []*model
}

var _ face.LivenessDetector = (*Detector)(nil)

/*
NewDetector loads the two ensemble models (v2Path: the "2.7_80x80_..."
scale-2.7 MiniFASNetV2 model; v1sePath: the "4_0_0_80x80_..." scale-4.0
MiniFASNetV1SE model -- see DownloadModel for how to obtain both).
InitEnvironment must have been called first.
*/
func NewDetector(v2Path, v1sePath string) (*Detector, error) {

	d := &Detector{}

	specs := []struct {
		path  string
		scale float64
	}{
		{v2Path, 2.7},
		{v1sePath, 4.0},
	}

	for _, spec := range specs {
		m, err := newModel(spec.path, spec.scale)
		if err != nil {
			d.Close()
			return nil, fmt.Errorf("loading %s: %w", spec.path, err)
		}
		d.models = append(d.models, m)
	}

	return d, nil

}

func newModel(path string, scale float64) (*model, error) {

	m := &model{scale: scale}

	var err error
	m.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, cropSize, cropSize))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}
	m.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3))
	if err != nil {
		m.input.Destroy()
		return nil, fmt.Errorf("allocating output tensor: %w", err)
	}

	m.session, err = ort.NewAdvancedSession(
		path,
		[]string{"input"},
		[]string{"output"},
		[]ort.Value{m.input},
		[]ort.Value{m.output},
		nil,
	)
	if err != nil {
		m.input.Destroy()
		m.output.Destroy()
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return m, nil

}

// Close releases the resources held by the detector.
func (d *Detector) Close() {
	for _, m := range d.models {
		if m.session != nil {
			m.session.Destroy()
		}
		if m.input != nil {
			m.input.Destroy()
		}
		if m.output != nil {
			m.output.Destroy()
		}
	}
}

/*
Detect classifies f (a face from any detector -- see the package doc)
in img as live or a print/replay spoof. Only f.Rectangle is used.
*/
func (d *Detector) Detect(img image.Image, f face.Face) (Result, error) {

	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()

	var sum [3]float64
	for _, m := range d.models {

		cropRect := cropBox(srcW, srcH, f.Rectangle, m.scale)
		patch := cropAndResize(img, bounds.Min, cropRect)

		data := m.input.GetData()
		frameSize := cropSize * cropSize
		for y := 0; y < cropSize; y++ {
			for x := 0; x < cropSize; x++ {
				r, g, b, _ := patch.At(x, y).RGBA()
				idx := y*cropSize + x
				// Silent-Face-Anti-Spoofing's ToTensor: HWC(BGR,uint8)
				// -> CHW float32, deliberately with NO /255
				// normalization -- confirmed from the actual
				// src/data_io/functional.py source, where the
				// "return img.float().div(255)" line is commented out
				// and replaced with a bare "return img.float()".
				data[0*frameSize+idx] = float32(b >> 8)
				data[1*frameSize+idx] = float32(g >> 8)
				data[2*frameSize+idx] = float32(r >> 8)
			}
		}

		if err := m.session.Run(); err != nil {
			return Result{}, fmt.Errorf("running session: %w", err)
		}

		probs := softmax3(m.output.GetData())
		sum[0] += probs[0]
		sum[1] += probs[1]
		sum[2] += probs[2]

	}

	label := 0
	for i := 1; i < 3; i++ {
		if sum[i] > sum[label] {
			label = i
		}
	}

	return Result{
		IsLive: label == 1,
		Score:  sum[label] / float64(len(d.models)),
	}, nil

}

/*
cropBox replicates CropImage._get_new_box: an expanded (by scale),
roughly-square region centered on bbox's own center, clamped to fit
within a srcW x srcH image by shifting (not just clipping) if it would
otherwise overflow an edge -- so the returned rectangle is always
exactly the box that would be cropped, never smaller. Returned as a
half-open image.Rectangle (the original algorithm's inclusive integer
bounds, x2/y2, become Max.X/Max.Y+1 here).
*/
func cropBox(srcW, srcH int, bbox image.Rectangle, scale float64) image.Rectangle {

	x := float64(bbox.Min.X)
	y := float64(bbox.Min.Y)
	boxW := float64(bbox.Dx())
	boxH := float64(bbox.Dy())

	s := math.Min(float64(srcH-1)/boxH, math.Min(float64(srcW-1)/boxW, scale))

	newWidth := boxW * s
	newHeight := boxH * s
	centerX := boxW/2 + x
	centerY := boxH/2 + y

	leftTopX := centerX - newWidth/2
	leftTopY := centerY - newHeight/2
	rightBottomX := centerX + newWidth/2
	rightBottomY := centerY + newHeight/2

	if leftTopX < 0 {
		rightBottomX -= leftTopX
		leftTopX = 0
	}
	if leftTopY < 0 {
		rightBottomY -= leftTopY
		leftTopY = 0
	}
	if rightBottomX > float64(srcW-1) {
		leftTopX -= rightBottomX - float64(srcW-1)
		rightBottomX = float64(srcW - 1)
	}
	if rightBottomY > float64(srcH-1) {
		leftTopY -= rightBottomY - float64(srcH-1)
		rightBottomY = float64(srcH - 1)
	}

	x1, y1 := int(leftTopX), int(leftTopY)
	x2, y2 := int(rightBottomX), int(rightBottomY)

	return image.Rect(x1, y1, x2+1, y2+1)

}

// cropAndResize bilinear-resizes the region of img at rect (translated
// into img's own coordinate space via origin, since rect is computed in
// a 0,0-based space but img.Bounds() may not start at 0,0) into a
// cropSize x cropSize image, matching cv2.resize's default INTER_LINEAR.
func cropAndResize(img image.Image, origin image.Point, rect image.Rectangle) *image.RGBA {

	src := rect.Add(origin)
	out := image.NewRGBA(image.Rect(0, 0, cropSize, cropSize))
	draw.BiLinear.Scale(out, out.Bounds(), img, src, draw.Src, nil)
	return out

}

func softmax3(logits []float32) [3]float64 {

	max := math.Inf(-1)
	for _, v := range logits[:3] {
		if float64(v) > max {
			max = float64(v)
		}
	}

	var sum float64
	var exp [3]float64
	for i := 0; i < 3; i++ {
		exp[i] = math.Exp(float64(logits[i]) - max)
		sum += exp[i]
	}
	for i := range exp {
		exp[i] /= sum
	}

	return exp

}
