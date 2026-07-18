// Package centerface implements face detection using CenterFace
// (centerface.onnx from https://github.com/Star-Clouds/CenterFace, MIT
// licensed, trained on WIDER FACE). Detector implements the
// face.FaceDetector contract -- an alternative to yunet.Detector.
package centerface

import (
	"fmt"
	"image"
	"math"
	"sort"

	"golang.org/x/image/draw"

	"github.com/leandroveronezi/go-onnxface/face"
	ort "github.com/yalue/onnxruntime_go"
)

var _ face.FaceDetector = (*Detector)(nil)

// Detector runs CenterFace face detection. Unlike yunet.Detector,
// CenterFace's ONNX graph is fully convolutional with a dynamic input
// size (once fixed -- see the note in NewDetector), so Detect resizes to
// whatever size the input image needs rather than a fixed canvas.
type Detector struct {
	session *ort.DynamicAdvancedSession

	// ScoreThreshold is the minimum heatmap confidence to keep a
	// candidate detection. Default 0.5.
	ScoreThreshold float32
	// NMSThreshold is the IoU threshold above which a lower-scoring
	// candidate is suppressed by a higher-scoring one. Default 0.3.
	NMSThreshold float32
}

/*
NewDetector loads the CenterFace face detection model from modelPath.
Init must have been called first.

The model file must have a dynamic (not fixed-batch/fixed-size) input --
the ONNX file Star-Clouds publishes declares a fixed [10,3,32,32] input
(a leftover of how it was traced out of PyTorch without dynamic_axes),
which ONNX Runtime enforces strictly and OpenCV's cv::dnn -- what
CenterFace's own reference implementation uses -- does not. Relax it
first with:

	import onnx
	m = onnx.load("centerface.onnx")
	for t in (m.graph.input[0], *m.graph.output):
	    dims = t.type.tensor_type.shape.dim
	    dims[0].dim_param, dims[2].dim_param, dims[3].dim_param = "batch", "h", "w"
	    for i in (0, 2, 3):
	        dims[i].ClearField("dim_value")
	onnx.save(m, "centerface_dynamic.onnx")

This only edits shape metadata, not the trained weights.
*/
func NewDetector(modelPath string) (*Detector, error) {

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input.1"},
		[]string{"537", "538", "539", "540"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return &Detector{
		session:        session,
		ScoreThreshold: 0.5,
		NMSThreshold:   0.3,
	}, nil

}

// Close releases the resources held by the detector.
func (d *Detector) Close() {
	if d.session != nil {
		d.session.Destroy()
	}
}

// candidate is a detection in the network's own resized-input coordinate
// space, before mapping back to img's coordinate space.
type candidate struct {
	rect      [4]float64 // x1, y1, x2, y2
	landmarks [5][2]float64
	score     float32
}

/*
Detect finds faces in img, returning their rectangles/landmarks in img's
own coordinate space.
*/
func (d *Detector) Detect(img image.Image) ([]face.Face, error) {

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w == 0 || h == 0 {
		return nil, fmt.Errorf("empty image")
	}

	// CenterFace resizes (stretches, not letterboxes) to the input size
	// rounded up to a multiple of 32, tracking each axis' scale
	// separately since the resize isn't uniform.
	hNew := int(math.Ceil(float64(h)/32)) * 32
	wNew := int(math.Ceil(float64(w)/32)) * 32
	scaleH := float64(hNew) / float64(h)
	scaleW := float64(wNew) / float64(w)

	resized := image.NewRGBA(image.Rect(0, 0, wNew, hNew))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, bounds, draw.Src, nil)

	input, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, int64(hNew), int64(wNew)))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}
	defer input.Destroy()

	data := input.GetData()
	frameSize := hNew * wNew
	for y := 0; y < hNew; y++ {
		for x := 0; x < wNew; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			idx := y*wNew + x
			// blobFromImage(..., scalefactor=1, mean=(0,0,0),
			// swapRB=true, crop=false): RGB order, raw 0-255 values, no
			// normalization.
			data[0*frameSize+idx] = float32(r >> 8)
			data[1*frameSize+idx] = float32(g >> 8)
			data[2*frameSize+idx] = float32(b >> 8)
		}
	}

	outputs := []ort.Value{nil, nil, nil, nil}
	if err := d.session.Run([]ort.Value{input}, outputs); err != nil {
		return nil, fmt.Errorf("running session: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	heatmapT, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for heatmap output")
	}
	scaleT, ok := outputs[1].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for scale output")
	}
	offsetT, ok := outputs[2].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for offset output")
	}
	lmT, ok := outputs[3].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for landmark output")
	}

	hmShape := heatmapT.GetShape()
	hs := int(hmShape[2])
	ws := int(hmShape[3])
	grid := hs * ws

	heatmap := heatmapT.GetData()
	scaleData := scaleT.GetData()
	offsetData := offsetT.GetData()
	lmData := lmT.GetData()

	var candidates []candidate
	for r := 0; r < hs; r++ {
		for c := 0; c < ws; c++ {
			idx := r*ws + c

			score := heatmap[idx]
			if score <= d.ScoreThreshold {
				continue
			}

			// scale/offset channel 0 is height (row-axis), channel 1 is
			// width (column-axis) -- matching centerface.py's
			// scale0/scale1, offset0/offset1 split.
			s0 := math.Exp(float64(scaleData[0*grid+idx])) * 4
			s1 := math.Exp(float64(scaleData[1*grid+idx])) * 4
			o0 := float64(offsetData[0*grid+idx])
			o1 := float64(offsetData[1*grid+idx])

			cx := (float64(c) + o1 + 0.5) * 4
			cy := (float64(r) + o0 + 0.5) * 4

			x1 := math.Max(0, cx-s1/2)
			y1 := math.Max(0, cy-s0/2)
			x1 = math.Min(x1, float64(wNew))
			y1 = math.Min(y1, float64(hNew))
			x2 := math.Min(x1+s1, float64(wNew))
			y2 := math.Min(y1+s0, float64(hNew))

			var landmarks [5][2]float64
			for j := 0; j < 5; j++ {
				// Channels are interleaved (y,x) per point: channel 2j
				// is the y-component (scaled by height, s0), channel
				// 2j+1 is the x-component (scaled by width, s1).
				lx := float64(lmData[(j*2+1)*grid+idx])*s1 + x1
				ly := float64(lmData[(j*2)*grid+idx])*s0 + y1
				landmarks[j] = [2]float64{lx, ly}
			}

			candidates = append(candidates, candidate{
				rect:      [4]float64{x1, y1, x2, y2},
				landmarks: landmarks,
				score:     score,
			})

		}
	}

	kept := nms(candidates, d.NMSThreshold)

	faces := make([]face.Face, len(kept))
	for i, cand := range kept {

		var landmarks [5]image.Point
		for j, lm := range cand.landmarks {
			landmarks[j] = image.Point{
				X: int(math.Round(lm[0] / scaleW)),
				Y: int(math.Round(lm[1] / scaleH)),
			}
		}

		faces[i] = face.Face{
			Rectangle: image.Rect(
				int(math.Round(cand.rect[0]/scaleW)),
				int(math.Round(cand.rect[1]/scaleH)),
				int(math.Round(cand.rect[2]/scaleW)),
				int(math.Round(cand.rect[3]/scaleH)),
			),
			Landmarks: landmarks,
			Score:     cand.score,
		}

	}

	return faces, nil

}

// nms performs greedy non-maximum suppression matching centerface.py's
// own nms(): sort by score descending, suppress a candidate whose IoU
// against any higher-scoring kept candidate is at or above threshold,
// using the (x2-x1+1)*(y2-y1+1) area convention centerface.py uses.
func nms(candidates []candidate, iouThreshold float32) []candidate {

	sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })

	kept := make([]candidate, 0, len(candidates))
	for _, cand := range candidates {
		overlaps := false
		for _, k := range kept {
			if iou(cand.rect, k.rect) >= iouThreshold {
				overlaps = true
				break
			}
		}
		if !overlaps {
			kept = append(kept, cand)
		}
	}

	return kept

}

func iou(a, b [4]float64) float32 {

	x1 := math.Max(a[0], b[0])
	y1 := math.Max(a[1], b[1])
	x2 := math.Min(a[2], b[2])
	y2 := math.Min(a[3], b[3])

	iw := math.Max(0, x2-x1+1)
	ih := math.Max(0, y2-y1+1)
	inter := iw * ih

	areaA := (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
	areaB := (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}

	return float32(inter / union)

}
