/*
Package seetaface6 implements SeetaFace6's liveness module (fas_first +
fas_second, an unofficial ONNX conversion of the original
fas_first.csta/fas_second.csta -- SeetaFace6 itself only ships the
proprietary .csta format, no ONNX). BSD-licensed
(https://github.com/seetafaceengine/SeetaFace6), trained for this exact
task, same licensing situation as the liveness package.

Detector implements face.LivenessDetector, the same contract
liveness.Detector does -- both can be plugged into the easy Recognizer
API via LivenessEngine. The two disagree on a real precision/recall
trade-off, not just accuracy: SeetaFace6 catches print/replay spoofs far
more aggressively than MiniFASNet, at the cost of rejecting far more
real people (see the go-onnxface-benchmarks README for the numbers this
was validated against).

Algorithm ported directly from
FaceAntiSpoofingX/src/seeta/FaceAntiSpoofing.cpp (not guessed):

 1. fas_second is a full-image SSD detector (MobileNetV2 backbone, 1917
    fixed anchors -- see anchors.go) that looks for spoof-indicator
    objects (screen bezels, paper edges, etc) anywhere in the frame. If
    it finds ANY box above class_threshold=0.8 after NMS(threshold=0.8),
    the image is SPOOF immediately -- fas_first is never even called.
    Preprocessing (confirmed from source, not guessed): resize to
    300x300, (pixel/127.5)-1.0, RGB.
 2. If fas_second found nothing, fas_first classifies the aligned face
    crop: convert to YCrCb (exact fixed-point formula from
    vipl_BGR2YCrCb, not cv2's own conversion), resize to 224x224 (the
    real network input size, confirmed both mathematically -- the
    AveragePool kernel_shape is 7x7 and the backbone downsamples 32x, so
    224/32=7 is exactly required -- and empirically against real
    photos), fed as raw [0,255] pixel values (same "no normalization at
    the ONNX boundary" pattern as face_recognizer.onnx). Output index 1
    is the "real" probability (the graph's output is literally named
    "prob_51", implying softmax is already baked in). fuseThreshold=0.8
    decides real vs spoof.

Deliberately NOT ported: FaceAntiSpoofing.cpp's ClarityEstimate (a
classical re-blur/sharpness heuristic) only distinguishes the original
SDK's REAL from FUZZY -- both of which are "not spoof" -- it never
affects the actual spoof/not-spoof decision, which depends only on
fas_first/fas_second above.
*/
package seetaface6

import (
	"fmt"
	"image"
	"math"
	"sort"

	"github.com/leandroveronezi/go-onnxface/face"
	ort "github.com/yalue/onnxruntime_go"
)

var _ face.LivenessDetector = (*Detector)(nil)

// ---- fas_second: full-image SSD spoof-object detector ----

const fasSecondSize = 300

type fasSecond struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	scores  *ort.Tensor[float32]
	boxes   *ort.Tensor[float32]
}

func newFasSecond(modelPath string) (*fasSecond, error) {

	d := &fasSecond{}

	var err error
	d.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, fasSecondSize, fasSecondSize))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}
	d.scores, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1917, 3, 1))
	if err != nil {
		d.input.Destroy()
		return nil, fmt.Errorf("allocating scores tensor: %w", err)
	}
	d.boxes, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1917, 1, 4))
	if err != nil {
		d.input.Destroy()
		d.scores.Destroy()
		return nil, fmt.Errorf("allocating boxes tensor: %w", err)
	}

	d.session, err = ort.NewAdvancedSession(
		modelPath,
		[]string{"_input_151"},
		[]string{"convert_scores_589", "concat_673"},
		[]ort.Value{d.input},
		[]ort.Value{d.scores, d.boxes},
		nil,
	)
	if err != nil {
		d.input.Destroy()
		d.scores.Destroy()
		d.boxes.Destroy()
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return d, nil

}

func (d *fasSecond) Close() {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.input != nil {
		d.input.Destroy()
	}
	if d.scores != nil {
		d.scores.Destroy()
	}
	if d.boxes != nil {
		d.boxes.Destroy()
	}
}

const (
	classThreshold = 0.8
	nmsThreshold   = 0.8
)

// hasBox reports whether fas_second found any spoof-indicator object
// anywhere in img (a signal that, on its own, means SPOOF).
func (d *fasSecond) hasBox(img image.Image) (bool, error) {

	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()

	resized := image.NewRGBA(image.Rect(0, 0, fasSecondSize, fasSecondSize))
	// nearest-neighbor is fine here -- this is a coarse full-image
	// object detector, not a precision alignment step.
	for y := 0; y < fasSecondSize; y++ {
		sy := b.Min.Y + y*srcH/fasSecondSize
		for x := 0; x < fasSecondSize; x++ {
			sx := b.Min.X + x*srcW/fasSecondSize
			resized.Set(x, y, img.At(sx, sy))
		}
	}

	data := d.input.GetData()
	frameSize := fasSecondSize * fasSecondSize
	for y := 0; y < fasSecondSize; y++ {
		for x := 0; x < fasSecondSize; x++ {
			rr, gg, bb, _ := resized.At(x, y).RGBA()
			idx := y*fasSecondSize + x
			data[0*frameSize+idx] = float32(rr>>8)*0.00784313771874 - 1.0
			data[1*frameSize+idx] = float32(gg>>8)*0.00784313771874 - 1.0
			data[2*frameSize+idx] = float32(bb>>8)*0.00784313771874 - 1.0
		}
	}

	if err := d.session.Run(); err != nil {
		return false, fmt.Errorf("running session: %w", err)
	}

	scores := d.scores.GetData() // [1917][3]
	boxes := d.boxes.GetData()   // [1917][4], (dy,dx,dh,dw)-style encoding per decodeBox

	type box struct {
		score          float32
		area           float64
		x1, y1, x2, y2 float64
	}
	var candidates []box

	for i := 0; i < 1917; i++ {

		s1, s2 := scores[i*3+1], scores[i*3+2]
		score := s1
		if s2 > s1 {
			score = s2
		}
		if score < classThreshold {
			continue
		}

		anchor := anchors[i]
		enc := [4]float64{
			float64(boxes[i*4+0]), float64(boxes[i*4+1]),
			float64(boxes[i*4+2]), float64(boxes[i*4+3]),
		}
		x1, y1, x2, y2 := decodeBox(anchor, enc)

		candidates = append(candidates, box{
			score: score,
			area:  (x2 - x1) * (y2 - y1),
			x1:    x1, y1: y1, x2: x2, y2: y2,
		})

	}

	if len(candidates) == 0 {
		return false, nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].area > candidates[j].area
	})

	suppressed := make([]bool, len(candidates))
	for i := range candidates {
		if suppressed[i] {
			continue
		}
		for j := i + 1; j < len(candidates); j++ {
			if iou(candidates[i], candidates[j]) > nmsThreshold {
				suppressed[j] = true
			}
		}
	}

	for i := range candidates {
		if !suppressed[i] {
			return true, nil
		}
	}
	return false, nil

}

type ssdBox = struct {
	score          float32
	area           float64
	x1, y1, x2, y2 float64
}

func iou(a, b ssdBox) float64 {
	ix1, iy1 := math.Max(a.x1, b.x1), math.Max(a.y1, b.y1)
	ix2, iy2 := math.Min(a.x2, b.x2), math.Min(a.y2, b.y2)
	iw, ih := math.Max(0, ix2-ix1), math.Max(0, iy2-iy1)
	inter := iw * ih
	union := a.area + b.area - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// decodeBox replicates FaceAntiSpoofing.cpp's decode_box: swap
// anchor/encoding coordinate order, then standard SSD center-size decode
// with variances (0.1, 0.1, 0.2, 0.2). Returns normalized [0,1] x1,y1,x2,y2.
func decodeBox(anchor [4]float64, enc [4]float64) (x1, y1, x2, y2 float64) {

	a := [4]float64{anchor[1], anchor[0], anchor[3], anchor[2]}
	e := [4]float64{enc[1], enc[0], enc[3], enc[2]}

	width := a[2] - a[0]
	height := a[3] - a[1]
	ctrX := a[0] + 0.5*width
	ctrY := a[1] + 0.5*height

	predCtrX := e[0]*0.1*width + ctrX
	predCtrY := e[1]*0.1*height + ctrY
	predW := math.Exp(e[2]*0.2) * width
	predH := math.Exp(e[3]*0.2) * height

	return predCtrX - 0.5*predW, predCtrY - 0.5*predH, predCtrX + 0.5*predW, predCtrY + 0.5*predH

}

// ---- fas_first: aligned-face-crop real/spoof classifier ----

const fasFirstSize = 224

// fasFirstMeanShape is FaceAntiSpoofing.cpp's own 512-canvas mean_shape
// (the 256-canvas recognizer template + cropFaceSize/4 = 128 offset on
// both axes).
var fasFirstMeanShape = [5][2]float64{
	{89.3095 + 128, 72.9025 + 128},
	{169.3095 + 128, 72.9025 + 128},
	{127.8949 + 128, 127.0441 + 128},
	{96.8796 + 128, 184.8907 + 128},
	{159.1065 + 128, 184.7601 + 128},
}

type fasFirst struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

func newFasFirst(modelPath string) (*fasFirst, error) {

	d := &fasFirst{}

	var err error
	d.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, fasFirstSize, fasFirstSize))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}
	d.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 4, 1, 1))
	if err != nil {
		d.input.Destroy()
		return nil, fmt.Errorf("allocating output tensor: %w", err)
	}

	d.session, err = ort.NewAdvancedSession(
		modelPath,
		[]string{"_input_8"},
		[]string{"prob_51"},
		[]ort.Value{d.input},
		[]ort.Value{d.output},
		nil,
	)
	if err != nil {
		d.input.Destroy()
		d.output.Destroy()
		return nil, fmt.Errorf("creating session: %w", err)
	}

	return d, nil

}

func (d *fasFirst) Close() {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.input != nil {
		d.input.Destroy()
	}
	if d.output != nil {
		d.output.Destroy()
	}
}

// score crops+aligns img to the 512-canvas mean_shape, converts to
// YCrCb, resizes to 224x224, and returns prob[1] (the "real" channel).
func (d *fasFirst) score(img image.Image, landmarks [5]image.Point) (float64, error) {

	var src [5][2]float64
	for i, p := range landmarks {
		src[i] = [2]float64{float64(p.X), float64(p.Y)}
	}
	t := estimateSimilarity(src, fasFirstMeanShape)
	crop512 := warpAffineTo(img, t, 512)

	data := d.input.GetData()
	frameSize := fasFirstSize * fasFirstSize
	scale := 512.0 / float64(fasFirstSize)
	for y := 0; y < fasFirstSize; y++ {
		sy := int(float64(y) * scale)
		for x := 0; x < fasFirstSize; x++ {
			sx := int(float64(x) * scale)
			rr, gg, bb, _ := crop512.At(sx, sy).RGBA()
			yy, cr, cb := bgr2YCrCb(float64(bb>>8), float64(gg>>8), float64(rr>>8))
			idx := y*fasFirstSize + x
			data[0*frameSize+idx] = float32(yy)
			data[1*frameSize+idx] = float32(cr)
			data[2*frameSize+idx] = float32(cb)
		}
	}

	if err := d.session.Run(); err != nil {
		return 0, fmt.Errorf("running session: %w", err)
	}

	out := d.output.GetData()
	return float64(out[1]), nil

}

// bgr2YCrCb replicates vipl_BGR2YCrCb's exact fixed-point formula (not
// cv2's own, which rounds slightly differently). Returns (Y, Cr, Cb) in
// that order, matching how FaceAntiSpoofing.cpp stores them
// (YCrCb_data[0]=Y, [1]=V(=Cr), [2]=U(=Cb)).
func bgr2YCrCb(b, g, r float64) (y, cr, cb float64) {
	yy := (b*1868 + g*9617 + r*4899 + 8192) / 16384
	u := (b-yy)*9241/16384 + 8192/16384 + 128
	v := (r-yy)*11682/16384 + 8192/16384 + 128
	clamp := func(v float64) float64 {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return v
	}
	return clamp(yy), clamp(v), clamp(u)
}

// estimateSimilarity fits a similarity transform (rotation + uniform
// scale + translation) mapping src onto dst, both 5-point sets. A
// simpler closed-form fit than face.AlignCrop's Umeyama-based one
// (approximate scale via RMS distance from centroid, not SVD) --
// kept as its own implementation since it's validated against this
// exact model's mean_shape convention, not SFace/ArcFace's.
func estimateSimilarity(src, dst [5][2]float64) [2][3]float64 {

	var srcMean, dstMean [2]float64
	for i := 0; i < 5; i++ {
		srcMean[0] += src[i][0]
		srcMean[1] += src[i][1]
		dstMean[0] += dst[i][0]
		dstMean[1] += dst[i][1]
	}
	srcMean[0] /= 5
	srcMean[1] /= 5
	dstMean[0] /= 5
	dstMean[1] /= 5

	var num, den float64
	for i := 0; i < 5; i++ {
		sx, sy := src[i][0]-srcMean[0], src[i][1]-srcMean[1]
		dx, dy := dst[i][0]-dstMean[0], dst[i][1]-dstMean[1]
		num += dx*sy - dy*sx
		den += dx*sx + dy*sy
	}
	theta := 0.0
	if den != 0 || num != 0 {
		theta = -math.Atan2(num, den)
	}

	var srcRMS, dstRMS float64
	for i := 0; i < 5; i++ {
		sx, sy := src[i][0]-srcMean[0], src[i][1]-srcMean[1]
		dx, dy := dst[i][0]-dstMean[0], dst[i][1]-dstMean[1]
		srcRMS += sx*sx + sy*sy
		dstRMS += dx*dx + dy*dy
	}
	scale := math.Sqrt(dstRMS/5) / math.Sqrt(srcRMS/5)

	ct, st := math.Cos(theta), math.Sin(theta)
	r00, r01 := ct*scale, -st*scale
	r10, r11 := st*scale, ct*scale

	t02 := dstMean[0] - (r00*srcMean[0] + r01*srcMean[1])
	t12 := dstMean[1] - (r10*srcMean[0] + r11*srcMean[1])

	return [2][3]float64{
		{r00, r01, t02},
		{r10, r11, t12},
	}

}

func warpAffineTo(img image.Image, t [2][3]float64, size int) *image.RGBA {

	det := t[0][0]*t[1][1] - t[0][1]*t[1][0]
	inv00 := t[1][1] / det
	inv01 := -t[0][1] / det
	inv10 := -t[1][0] / det
	inv11 := t[0][0] / det

	out := image.NewRGBA(image.Rect(0, 0, size, size))
	bounds := img.Bounds()

	for v := 0; v < size; v++ {
		for u := 0; u < size; u++ {
			dx := float64(u) - t[0][2]
			dy := float64(v) - t[1][2]
			sx := inv00*dx + inv01*dy
			sy := inv10*dx + inv11*dy

			srcX, srcY := bounds.Min.X+int(sx), bounds.Min.Y+int(sy)
			if srcX < bounds.Min.X || srcX >= bounds.Max.X || srcY < bounds.Min.Y || srcY >= bounds.Max.Y {
				continue
			}
			out.Set(u, v, img.At(srcX, srcY))
		}
	}

	return out

}

// ---- fusion ----

const fuseThreshold = 0.8

// Detector runs the fas_first+fas_second liveness fusion. Implements
// face.LivenessDetector.
type Detector struct {
	first  *fasFirst
	second *fasSecond
}

// NewDetector loads both models (see DownloadModel for how to obtain
// them). InitEnvironment must have been called first.
func NewDetector(fasFirstPath, fasSecondPath string) (*Detector, error) {

	first, err := newFasFirst(fasFirstPath)
	if err != nil {
		return nil, fmt.Errorf("loading %s: %w", fasFirstPath, err)
	}
	second, err := newFasSecond(fasSecondPath)
	if err != nil {
		first.Close()
		return nil, fmt.Errorf("loading %s: %w", fasSecondPath, err)
	}
	return &Detector{first: first, second: second}, nil

}

// Close releases the resources held by both underlying models.
func (d *Detector) Close() {
	d.first.Close()
	d.second.Close()
}

/*
Detect classifies f (a face from any detector) in img as live or a
print/replay spoof. Only f.Landmarks is used -- fas_second runs on the
whole image (img), fas_first aligns from the landmarks; f.Rectangle is
ignored. Score is 0 when fas_second's full-image gate already decided
SPOOF on its own (matching the original SDK, where fas_first's
score_face is never called in that case).
*/
func (d *Detector) Detect(img image.Image, f face.Face) (face.Result, error) {

	hasBox, err := d.second.hasBox(img)
	if err != nil {
		return face.Result{}, err
	}
	if hasBox {
		return face.Result{IsLive: false, Score: 0}, nil
	}

	score, err := d.first.score(img, f.Landmarks)
	if err != nil {
		return face.Result{}, err
	}

	return face.Result{IsLive: score >= fuseThreshold, Score: score}, nil

}
