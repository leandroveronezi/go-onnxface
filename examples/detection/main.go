// Command detection demonstrates the low-level API with three
// interchangeable detectors -- yunet.Detector, centerface.Detector and
// retinaface.Detector -- all implementing face.FaceDetector, run side by
// side on the same image.
package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"

	"github.com/leandroveronezi/go-onnxface"
	"github.com/leandroveronezi/go-onnxface/centerface"
	"github.com/leandroveronezi/go-onnxface/retinaface"
	"github.com/leandroveronezi/go-onnxface/yunet"
)

const modelsDir = "../../models"

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return jpeg.Decode(f)
}

func detect(name string, det onnxface.FaceDetector, img image.Image) {
	faces, err := det.Detect(img)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s: %d face(s)\n", name, len(faces))
	for _, f := range faces {
		fmt.Printf("  box=%v score=%.4f\n", f.Rectangle, f.Score)
	}
}

func main() {

	ortLib := os.Getenv("ONNXFACE_ORT_LIB")
	if ortLib == "" {
		fmt.Println("set ONNXFACE_ORT_LIB to the onnxruntime shared library path (see the README)")
		return
	}
	if err := onnxface.InitEnvironment(ortLib); err != nil {
		fmt.Println(err)
		return
	}
	defer onnxface.CloseEnvironment()

	img, err := loadImage("../fotos/elenco3.jpg")
	if err != nil {
		fmt.Println(err)
		return
	}

	yn, err := yunet.NewDetector(modelsDir + "/face_detection_yunet_2023mar.onnx")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer yn.Close()

	cf, err := centerface.NewDetector(modelsDir + "/centerface.onnx")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cf.Close()

	rf, err := retinaface.NewDetector(modelsDir + "/retinaface.onnx")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer rf.Close()

	detect("YuNet", yn, img)
	detect("CenterFace", cf, img)
	detect("RetinaFace", rf, img)

}
