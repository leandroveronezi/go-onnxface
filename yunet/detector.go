// Package yunet implements face detection using YuNet
// (face_detection_yunet_2023mar.onnx from the OpenCV Zoo, MIT licensed).
// Detector implements the face.FaceDetector contract.
package yunet

import (
	"fmt"
	"image"
	"math"
	"sort"

	"golang.org/x/image/draw"

	"github.com/leandroveronezi/go-onnxface/face"
	ort "github.com/yalue/onnxruntime_go"
)

// yunetSize is the fixed input resolution the published YuNet ONNX graph
// (face_detection_yunet_2023mar.onnx) expects; unlike OpenCV's
// FaceDetectorYN wrapper (which can resize the underlying graph via
// setInputSize), the raw ONNX file has a static [1,3,640,640] input.
const yunetSize = 640

var yunetStrides = [3]int{8, 16, 32}

var _ face.FaceDetector = (*Detector)(nil)

// Detector runs YuNet face detection.
type Detector struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	cls     [3]*ort.Tensor[float32]
	obj     [3]*ort.Tensor[float32]
	bbox    [3]*ort.Tensor[float32]
	kps     [3]*ort.Tensor[float32]

	ScoreThreshold float32
	NMSThreshold   float32
	TopK           int
}

/*
NewDetector loads the YuNet face detection model from modelPath (e.g.
face_detection_yunet_2023mar.onnx from the OpenCV Zoo). InitEnvironment must have
been called first.
*/
func NewDetector(modelPath string) (*Detector, error) {

	d := &Detector{
		ScoreThreshold: 0.6,
		NMSThreshold:   0.3,
		TopK:           5000,
	}

	var err error
	d.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, yunetSize, yunetSize))
	if err != nil {
		return nil, fmt.Errorf("allocating input tensor: %w", err)
	}

	inputNames := []string{"input"}
	outputNames := make([]string, 0, 12)
	outputs := make([]ort.Value, 0, 12)

	for i, stride := range yunetStrides {
		grid := yunetSize / stride
		n := int64(grid * grid)

		d.cls[i], err = ort.NewEmptyTensor[float32](ort.NewShape(1, n, 1))
		if err != nil {
			d.Close()
			return nil, fmt.Errorf("allocating cls_%d tensor: %w", stride, err)
		}
		d.obj[i], err = ort.NewEmptyTensor[float32](ort.NewShape(1, n, 1))
		if err != nil {
			d.Close()
			return nil, fmt.Errorf("allocating obj_%d tensor: %w", stride, err)
		}
		d.bbox[i], err = ort.NewEmptyTensor[float32](ort.NewShape(1, n, 4))
		if err != nil {
			d.Close()
			return nil, fmt.Errorf("allocating bbox_%d tensor: %w", stride, err)
		}
		d.kps[i], err = ort.NewEmptyTensor[float32](ort.NewShape(1, n, 10))
		if err != nil {
			d.Close()
			return nil, fmt.Errorf("allocating kps_%d tensor: %w", stride, err)
		}

		outputNames = append(outputNames, fmt.Sprintf("cls_%d", stride))
		outputs = append(outputs, d.cls[i])
	}
	for i, stride := range yunetStrides {
		outputNames = append(outputNames, fmt.Sprintf("obj_%d", stride))
		outputs = append(outputs, d.obj[i])
	}
	for i, stride := range yunetStrides {
		outputNames = append(outputNames, fmt.Sprintf("bbox_%d", stride))
		outputs = append(outputs, d.bbox[i])
	}
	for i, stride := range yunetStrides {
		outputNames = append(outputNames, fmt.Sprintf("kps_%d", stride))
		outputs = append(outputs, d.kps[i])
	}

	d.session, err = ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		[]ort.Value{d.input},
		outputs,
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
	for i := range yunetStrides {
		if d.cls[i] != nil {
			d.cls[i].Destroy()
		}
		if d.obj[i] != nil {
			d.obj[i].Destroy()
		}
		if d.bbox[i] != nil {
			d.bbox[i].Destroy()
		}
		if d.kps[i] != nil {
			d.kps[i].Destroy()
		}
	}

}

/*
Detect finds faces in img, returning their rectangles/landmarks in img's
own coordinate space (not the model's internal 640x640 space).
*/
func (d *Detector) Detect(img image.Image) ([]face.Face, error) {

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w == 0 || h == 0 {
		return nil, fmt.Errorf("empty image")
	}

	scale := math.Min(float64(yunetSize)/float64(w), float64(yunetSize)/float64(h))
	nw := int(math.Round(float64(w) * scale))
	nh := int(math.Round(float64(h) * scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}

	// Letterbox: bilinear-resize into the top-left corner of a 640x640
	// canvas (matching cv2.resize's default INTER_LINEAR), zero-pad the
	// rest (matching padWithDivisor's BORDER_CONSTANT zero padding).
	resized := image.NewRGBA(image.Rect(0, 0, nw, nh))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, bounds, draw.Src, nil)

	data := d.input.GetData()
	frameSize := yunetSize * yunetSize
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			idx := y*yunetSize + x
			// NCHW, channel order B,G,R (blobFromImage default: no
			// swapRB, and cv2.imread's Mat is natively BGR), raw pixel
			// values 0-255 as float32 (blobFromImage's default
			// scalefactor is 1.0 -- no /255 normalization).
			data[0*frameSize+idx] = float32(b >> 8)
			data[1*frameSize+idx] = float32(g >> 8)
			data[2*frameSize+idx] = float32(r >> 8)
		}
	}
	// The rest of `data` (outside the nh x nw region) is already zero
	// from NewEmptyTensor, matching the zero-padded region.

	if err := d.session.Run(); err != nil {
		return nil, fmt.Errorf("running session: %w", err)
	}

	var candidates []face.Face
	for i, stride := range yunetStrides {
		grid := yunetSize / stride
		cls := d.cls[i].GetData()
		obj := d.obj[i].GetData()
		bbox := d.bbox[i].GetData()
		kps := d.kps[i].GetData()

		for r := 0; r < grid; r++ {
			for c := 0; c < grid; c++ {
				idx := r*grid + c

				clsScore := clamp01(cls[idx])
				objScore := clamp01(obj[idx])
				score := float32(math.Sqrt(float64(clsScore * objScore)))
				if score < d.ScoreThreshold {
					continue
				}

				cx := (float64(c) + float64(bbox[idx*4+0])) * float64(stride)
				cy := (float64(r) + float64(bbox[idx*4+1])) * float64(stride)
				fw := math.Exp(float64(bbox[idx*4+2])) * float64(stride)
				fh := math.Exp(float64(bbox[idx*4+3])) * float64(stride)
				x1 := cx - fw/2
				y1 := cy - fh/2

				var landmarks [5]image.Point
				for n := 0; n < 5; n++ {
					lx := (float64(kps[idx*10+2*n]) + float64(c)) * float64(stride)
					ly := (float64(kps[idx*10+2*n+1]) + float64(r)) * float64(stride)
					landmarks[n] = image.Point{X: int(math.Round(lx / scale)), Y: int(math.Round(ly / scale))}
				}

				candidates = append(candidates, face.Face{
					Rectangle: image.Rect(
						int(math.Round(x1/scale)),
						int(math.Round(y1/scale)),
						int(math.Round((x1+fw)/scale)),
						int(math.Round((y1+fh)/scale)),
					),
					Landmarks: landmarks,
					Score:     score,
				})
			}
		}
	}

	return nms(candidates, d.NMSThreshold, d.TopK), nil

}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// nms performs greedy non-maximum suppression, matching OpenCV's
// dnn::NMSBoxes semantics (sort by score descending, suppress boxes with
// IoU above threshold against any higher-scoring kept box).
func nms(faces []face.Face, iouThreshold float32, topK int) []face.Face {

	sort.Slice(faces, func(i, j int) bool { return faces[i].Score > faces[j].Score })

	kept := make([]face.Face, 0, len(faces))
	for _, f := range faces {
		if len(kept) >= topK {
			break
		}
		overlaps := false
		for _, k := range kept {
			if iou(f.Rectangle, k.Rectangle) > iouThreshold {
				overlaps = true
				break
			}
		}
		if !overlaps {
			kept = append(kept, f)
		}
	}

	return kept

}

func iou(a, b image.Rectangle) float32 {

	inter := a.Intersect(b)
	if inter.Empty() {
		return 0
	}
	interArea := inter.Dx() * inter.Dy()
	unionArea := a.Dx()*a.Dy() + b.Dx()*b.Dy() - interArea
	if unionArea <= 0 {
		return 0
	}
	return float32(interArea) / float32(unionArea)

}
