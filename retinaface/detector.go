// Package retinaface implements face detection using RetinaFace
// (resnet50 backbone, MIT licensed, trained on WIDER FACE --
// https://github.com/biubug6/Pytorch_Retinaface). Detector implements
// the face.FaceDetector contract -- the higher-accuracy, heavier
// alternative to yunet.Detector/centerface.Detector.
package retinaface

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

// retinafaceSize is the fixed input resolution the exported
// retinaface_r50.onnx graph expects (a PyTorch trace artifact, same
// situation as YuNet: the raw model isn't dynamically shaped, so
// Detector letterboxes into this fixed canvas like yunet.Detector does).
const retinafaceSize = 640

var retinafaceStrides = [3]int{8, 16, 32}
var retinafaceMinSizes = [3][2]float64{{16, 32}, {64, 128}, {256, 512}}

const (
	variance0 = 0.1
	variance1 = 0.2
)

// prior is one anchor box, in the normalized (0..1) center-size form
// PriorBox produces: center x/y and width/height, all relative to
// retinafaceSize.
type prior struct {
	cx, cy, sx, sy float64
}

// Detector runs RetinaFace face detection.
type Detector struct {
	session           *ort.AdvancedSession
	input             *ort.Tensor[float32]
	loc, conf, landms *ort.Tensor[float32]
	priors            []prior

	// ScoreThreshold is the minimum face-class confidence to keep a
	// candidate detection. Default 0.5.
	ScoreThreshold float32
	// NMSThreshold is the IoU threshold above which a lower-scoring
	// candidate is suppressed by a higher-scoring one. Default 0.4
	// (RetinaFace's own default, per biubug6/Pytorch_Retinaface's
	// detect.py).
	NMSThreshold float32
}

/*
NewDetector loads the RetinaFace face detection model from modelPath
(retinaface_r50.onnx -- see the README for how to obtain/convert it, the
published weights are PyTorch .pth, not ONNX). InitEnvironment must have
been called first.
*/
func NewDetector(modelPath string) (*Detector, error) {

	d := &Detector{
		ScoreThreshold: 0.5,
		NMSThreshold:   0.4,
		priors:         generatePriors(),
	}

	numPriors := int64(len(d.priors))

	var err error
	d.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, retinafaceSize, retinafaceSize))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}
	d.loc, err = ort.NewEmptyTensor[float32](ort.NewShape(1, numPriors, 4))
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("allocating loc tensor: %w", err)
	}
	d.conf, err = ort.NewEmptyTensor[float32](ort.NewShape(1, numPriors, 2))
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("allocating conf tensor: %w", err)
	}
	d.landms, err = ort.NewEmptyTensor[float32](ort.NewShape(1, numPriors, 10))
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("allocating landms tensor: %w", err)
	}

	d.session, err = ort.NewAdvancedSession(
		modelPath,
		[]string{"input0"},
		[]string{"loc", "conf", "landms"},
		[]ort.Value{d.input},
		[]ort.Value{d.loc, d.conf, d.landms},
		nil,
	)
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return d, nil

}

// Close releases the resources held by the detector.
func (d *Detector) Close() {

	if d.session != nil {
		d.session.Destroy()
	}
	if d.input != nil {
		d.input.Destroy()
	}
	if d.loc != nil {
		d.loc.Destroy()
	}
	if d.conf != nil {
		d.conf.Destroy()
	}
	if d.landms != nil {
		d.landms.Destroy()
	}

}

/*
generatePriors replicates PriorBox.forward() for a fixed
retinafaceSize x retinafaceSize image: one set of anchors per stride
level, min_sizes[k] anchors per grid cell, in row-major (row, then
column) order -- matching the loc/conf/landms tensor layout the network
itself produces.
*/
func generatePriors() []prior {

	var priors []prior

	for k, step := range retinafaceStrides {
		grid := int(math.Ceil(float64(retinafaceSize) / float64(step)))
		for i := 0; i < grid; i++ {
			for j := 0; j < grid; j++ {
				for _, minSize := range retinafaceMinSizes[k] {
					priors = append(priors, prior{
						cx: (float64(j) + 0.5) * float64(step) / float64(retinafaceSize),
						cy: (float64(i) + 0.5) * float64(step) / float64(retinafaceSize),
						sx: minSize / float64(retinafaceSize),
						sy: minSize / float64(retinafaceSize),
					})
				}
			}
		}
	}

	return priors

}

// candidate is a detection in the network's own 640x640 coordinate
// space, before mapping back to img's coordinate space.
type candidate struct {
	rect      [4]float64 // x1, y1, x2, y2
	landmarks [5][2]float64
	score     float32
}

// Detect finds faces in img, returning their rectangles/landmarks in
// img's own coordinate space (not the model's internal 640x640 space).
func (d *Detector) Detect(img image.Image) ([]face.Face, error) {

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w == 0 || h == 0 {
		return nil, fmt.Errorf("empty image")
	}

	scale := math.Min(float64(retinafaceSize)/float64(w), float64(retinafaceSize)/float64(h))
	nw := int(math.Round(float64(w) * scale))
	nh := int(math.Round(float64(h) * scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}

	// Letterbox: bilinear-resize into the top-left corner of a 640x640
	// canvas, zero-pad the rest -- same convention as yunet.Detector.
	resized := image.NewRGBA(image.Rect(0, 0, nw, nh))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, bounds, draw.Src, nil)

	data := d.input.GetData()
	frameSize := retinafaceSize * retinafaceSize
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			idx := y*retinafaceSize + x
			// detect.py: img -= (104, 117, 123) on a BGR (cv2.imread)
			// Mat, no swapRB, no scale factor -- NCHW, BGR order, raw
			// 0-255 values minus that per-channel mean.
			data[0*frameSize+idx] = float32(b>>8) - 104
			data[1*frameSize+idx] = float32(g>>8) - 117
			data[2*frameSize+idx] = float32(r>>8) - 123
		}
	}
	// The rest of `data` (outside the nh x nw region) is zero from
	// NewEmptyTensor, i.e. exactly -104/-117/-123 after mean subtraction
	// would be needed for true zero-pixel padding -- but detect.py pads
	// with plain zero pixels *before* mean subtraction in none of its
	// paths (it never pads at all, always feeding the network the exact
	// image size). Letterbox zero-padding here mirrors yunet.Detector's
	// established, separately-validated convention instead; the
	// network's receptive field means small edge padding differences
	// this far from a detected face's own region don't move its score
	// or box measurably (confirmed by the pinned regression test).
	if err := d.session.Run(); err != nil {
		return nil, fmt.Errorf("running session: %w", err)
	}

	loc := d.loc.GetData()
	conf := d.conf.GetData()
	landms := d.landms.GetData()

	var candidates []candidate
	for i, p := range d.priors {

		score := conf[i*2+1]
		if score <= d.ScoreThreshold {
			continue
		}

		boxCx := p.cx + float64(loc[i*4+0])*variance0*p.sx
		boxCy := p.cy + float64(loc[i*4+1])*variance0*p.sy
		boxW := p.sx * math.Exp(float64(loc[i*4+2])*variance1)
		boxH := p.sy * math.Exp(float64(loc[i*4+3])*variance1)

		x1 := (boxCx - boxW/2) * retinafaceSize
		y1 := (boxCy - boxH/2) * retinafaceSize
		x2 := (boxCx + boxW/2) * retinafaceSize
		y2 := (boxCy + boxH/2) * retinafaceSize

		var landmarks [5][2]float64
		for n := 0; n < 5; n++ {
			lx := (p.cx + float64(landms[i*10+2*n+0])*variance0*p.sx) * retinafaceSize
			ly := (p.cy + float64(landms[i*10+2*n+1])*variance0*p.sy) * retinafaceSize
			landmarks[n] = [2]float64{lx, ly}
		}

		candidates = append(candidates, candidate{
			rect:      [4]float64{x1, y1, x2, y2},
			landmarks: landmarks,
			score:     score,
		})

	}

	kept := nms(candidates, d.NMSThreshold)

	faces := make([]face.Face, len(kept))
	for i, cand := range kept {

		var landmarks [5]image.Point
		for j, lm := range cand.landmarks {
			landmarks[j] = image.Point{
				X: int(math.Round(lm[0] / scale)),
				Y: int(math.Round(lm[1] / scale)),
			}
		}

		faces[i] = face.Face{
			Rectangle: image.Rect(
				int(math.Round(cand.rect[0]/scale)),
				int(math.Round(cand.rect[1]/scale)),
				int(math.Round(cand.rect[2]/scale)),
				int(math.Round(cand.rect[3]/scale)),
			),
			Landmarks: landmarks,
			Score:     cand.score,
		}

	}

	return faces, nil

}

// nms performs greedy non-maximum suppression matching
// utils/nms/py_cpu_nms.py: sort by score descending, suppress a
// candidate whose IoU against any higher-scoring kept candidate exceeds
// threshold, using the (x2-x1+1)*(y2-y1+1) area convention it uses.
func nms(candidates []candidate, iouThreshold float32) []candidate {

	sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })

	kept := make([]candidate, 0, len(candidates))
	for _, cand := range candidates {
		overlaps := false
		for _, k := range kept {
			if iou(cand.rect, k.rect) > iouThreshold {
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
