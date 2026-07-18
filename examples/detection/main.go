// Command detection demonstrates the low-level API with two
// interchangeable detectors -- yunet.Detector and centerface.Detector --
// both implementing face.FaceDetector, run side by side on the same
// image.
package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"

	"github.com/leandroveronezi/go-onnxface"
	"github.com/leandroveronezi/go-onnxface/centerface"
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

	ynFaces, err := yn.Detect(img)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("YuNet: %d face(s)\n", len(ynFaces))
	for _, f := range ynFaces {
		fmt.Printf("  box=%v score=%.4f\n", f.Rectangle, f.Score)
	}

	cfFaces, err := cf.Detect(img)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("CenterFace: %d face(s)\n", len(cfFaces))
	for _, f := range cfFaces {
		fmt.Printf("  box=%v score=%.4f\n", f.Rectangle, f.Score)
	}

}
