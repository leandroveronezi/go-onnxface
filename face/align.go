package face

import (
	"image"
	"image/color"
	"math"
)

// AlignedSize is the fixed crop size AlignCrop warps faces to.
const AlignedSize = 112

/*
standardTemplate is the canonical 112x112 landmark template (right eye,
left eye, nose tip, right corner of mouth, left corner of mouth) that
AlignCrop warps detected landmarks onto. Taken verbatim from OpenCV's
FaceRecognizerSFImpl::getSimilarityTransformMatrix -- and confirmed
identical (same 5 points) to InsightFace's own arcface_dst in
face_align.py, so this one template serves both SFace and the
ArcFace-family recognizer input convention.
*/
var standardTemplate = [5][2]float64{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}

// flt32Min mirrors C's FLT_MIN (smallest positive normalized float32),
// used by the rank check in similarityTransform.
const flt32Min = 1.1754943508222875e-38

/*
AlignCrop warps img to a canonical 112x112 face crop using the 5 detected
landmarks (as returned by a FaceDetector), matching
cv::FaceRecognizerSF::alignCrop (and equivalently InsightFace's
face_align.norm_crop): a similarity transform (rotation, scale,
translation -- no perspective) is estimated from the landmarks onto a
fixed template, then applied with bilinear interpolation.
*/
func AlignCrop(img image.Image, landmarks [5]image.Point) *image.RGBA {

	var src [5][2]float64
	for i, p := range landmarks {
		src[i] = [2]float64{float64(p.X), float64(p.Y)}
	}

	t := similarityTransform(src)
	return warpAffine(img, t, AlignedSize)

}

/*
similarityTransform estimates the 2x3 affine matrix mapping src (5
points) onto standardTemplate, via Umeyama's method. Ported line-by-line
from cv::FaceRecognizerSFImpl::getSimilarityTransformMatrix so it
produces the same matrix (rotation + uniform scale + translation, with a
reflection correction when needed) as OpenCV does.
*/
func similarityTransform(src [5][2]float64) [2][3]float64 {

	dst := standardTemplate

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

	var srcDemean, dstDemean [5][2]float64
	for i := 0; i < 5; i++ {
		srcDemean[i][0] = src[i][0] - srcMean[0]
		srcDemean[i][1] = src[i][1] - srcMean[1]
		dstDemean[i][0] = dst[i][0] - dstMean[0]
		dstDemean[i][1] = dst[i][1] - dstMean[1]
	}

	var a00, a01, a10, a11 float64
	for i := 0; i < 5; i++ {
		a00 += dstDemean[i][0] * srcDemean[i][0]
		a01 += dstDemean[i][0] * srcDemean[i][1]
		a10 += dstDemean[i][1] * srcDemean[i][0]
		a11 += dstDemean[i][1] * srcDemean[i][1]
	}
	a00 /= 5
	a01 /= 5
	a10 /= 5
	a11 /= 5

	u, s, vt := svd2x2(a00, a01, a10, a11)

	d0, d1 := 1.0, 1.0
	detA := a00*a11 - a01*a10
	if detA < 0 {
		d1 = -1
	}

	smax := s[0]
	if s[1] > smax {
		smax = s[1]
	}
	tol := smax * 2 * flt32Min
	rank := 0
	if s[0] > tol {
		rank++
	}
	if s[1] > tol {
		rank++
	}

	detU := u[0][0]*u[1][1] - u[0][1]*u[1][0]
	detVt := vt[0][0]*vt[1][1] - vt[0][1]*vt[1][0]

	mul2x2 := func(m1, m2 [2][2]float64) [2][2]float64 {
		return [2][2]float64{
			{m1[0][0]*m2[0][0] + m1[0][1]*m2[1][0], m1[0][0]*m2[0][1] + m1[0][1]*m2[1][1]},
			{m1[1][0]*m2[0][0] + m1[1][1]*m2[1][0], m1[1][0]*m2[0][1] + m1[1][1]*m2[1][1]},
		}
	}

	var tLinear [2][2]float64
	if rank == 1 {
		if detU*detVt > 0 {
			tLinear = mul2x2(u, vt)
		} else {
			d := [2][2]float64{{d0, 0}, {0, -1}}
			tLinear = mul2x2(u, mul2x2(d, vt))
		}
	} else {
		d := [2][2]float64{{d0, 0}, {0, d1}}
		tLinear = mul2x2(u, mul2x2(d, vt))
	}

	var var1, var2 float64
	for i := 0; i < 5; i++ {
		var1 += srcDemean[i][0] * srcDemean[i][0]
		var2 += srcDemean[i][1] * srcDemean[i][1]
	}
	var1 /= 5
	var2 /= 5

	scale := (1.0 / (var1 + var2)) * (s[0]*d0 + s[1]*d1)

	ts0 := tLinear[0][0]*srcMean[0] + tLinear[0][1]*srcMean[1]
	ts1 := tLinear[1][0]*srcMean[0] + tLinear[1][1]*srcMean[1]

	t02 := dstMean[0] - scale*ts0
	t12 := dstMean[1] - scale*ts1

	return [2][3]float64{
		{tLinear[0][0] * scale, tLinear[0][1] * scale, t02},
		{tLinear[1][0] * scale, tLinear[1][1] * scale, t12},
	}

}

/*
svd2x2 computes the SVD of the 2x2 matrix [[a,b],[c,d]] such that
A = U * diag(s0,s1) * Vt, with s0 >= s1 >= 0, matching the convention of
cv::SVD::compute(A, s, u, vt). V's columns are the eigenvectors of A^T*A
(a symmetric 2x2 matrix, solved in closed form); U's columns are then
A*v_i/s_i, falling back to an orthogonal completion when a singular value
is ~0.
*/
func svd2x2(a, b, c, d float64) (u [2][2]float64, s [2]float64, vt [2][2]float64) {

	m00 := a*a + c*c
	m11 := b*b + d*d
	m01 := a*b + c*d

	theta := 0.5 * math.Atan2(2*m01, m00-m11)
	ct, st := math.Cos(theta), math.Sin(theta)

	common := math.Hypot(m00-m11, 2*m01)
	lambda0 := math.Max((m00+m11)/2+common/2, 0)
	lambda1 := math.Max((m00+m11)/2-common/2, 0)
	s0 := math.Sqrt(lambda0)
	s1 := math.Sqrt(lambda1)

	v0 := [2]float64{ct, st}
	v1 := [2]float64{-st, ct}

	const eps = 1e-12
	var u0, u1 [2]float64
	haveU0 := s0 > eps
	haveU1 := s1 > eps
	if haveU0 {
		u0 = [2]float64{(a*v0[0] + b*v0[1]) / s0, (c*v0[0] + d*v0[1]) / s0}
	}
	if haveU1 {
		u1 = [2]float64{(a*v1[0] + b*v1[1]) / s1, (c*v1[0] + d*v1[1]) / s1}
	}
	switch {
	case !haveU0 && !haveU1:
		u0, u1 = [2]float64{1, 0}, [2]float64{0, 1}
	case !haveU0:
		u0 = [2]float64{-u1[1], u1[0]}
	case !haveU1:
		u1 = [2]float64{-u0[1], u0[0]}
	}

	u = [2][2]float64{{u0[0], u1[0]}, {u0[1], u1[1]}}
	vt = [2][2]float64{{v0[0], v0[1]}, {v1[0], v1[1]}}
	s = [2]float64{s0, s1}
	return

}

/*
warpAffine resamples img through the inverse of the 2x3 affine matrix t,
producing a size x size RGBA image, matching
cv::warpAffine(..., Size(size,size), INTER_LINEAR) with its default
BORDER_CONSTANT(0): out-of-bounds source samples contribute 0.
*/
func warpAffine(img image.Image, t [2][3]float64, size int) *image.RGBA {

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

			r, g, b, a := bilinearSample(img, bounds, sx, sy)
			out.SetRGBA(u, v, color.RGBA{R: r, G: g, B: b, A: a})
		}
	}

	return out

}

func sampleAt(img image.Image, b image.Rectangle, x, y int) (r, g, bl, a float64) {

	if x < b.Min.X || x >= b.Max.X || y < b.Min.Y || y >= b.Max.Y {
		return 0, 0, 0, 0
	}
	rr, gg, bb, aa := img.At(x, y).RGBA()
	return float64(rr >> 8), float64(gg >> 8), float64(bb >> 8), float64(aa >> 8)

}

func bilinearSample(img image.Image, b image.Rectangle, sx, sy float64) (r, g, bl, a uint8) {

	x0 := int(math.Floor(sx))
	y0 := int(math.Floor(sy))
	fx := sx - float64(x0)
	fy := sy - float64(y0)

	r00, g00, b00, a00 := sampleAt(img, b, x0, y0)
	r10, g10, b10, a10 := sampleAt(img, b, x0+1, y0)
	r01, g01, b01, a01 := sampleAt(img, b, x0, y0+1)
	r11, g11, b11, a11 := sampleAt(img, b, x0+1, y0+1)

	lerp := func(v00, v10, v01, v11 float64) float64 {
		top := v00 + (v10-v00)*fx
		bot := v01 + (v11-v01)*fx
		return top + (bot-top)*fy
	}

	clampByte := func(v float64) uint8 {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return uint8(math.Round(v))
	}

	return clampByte(lerp(r00, r10, r01, r11)),
		clampByte(lerp(g00, g10, g01, g11)),
		clampByte(lerp(b00, b10, b01, b11)),
		clampByte(lerp(a00, a10, a01, a11))

}
