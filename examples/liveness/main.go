// Command liveness demonstrates print/replay spoof detection: detect a
// face with any FaceDetector, then classify that same rectangle as live
// or a spoof.
package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"

	"github.com/leandroveronezi/go-onnxface"
	"github.com/leandroveronezi/go-onnxface/liveness"
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

	// Both ensemble models are always used together, so they're fetched
	// together too.
	if err := liveness.DownloadModel(modelsDir); err != nil {
		fmt.Println(err)
		return
	}

	det, err := yunet.NewDetector(modelsDir + "/face_detection_yunet_2023mar.onnx")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer det.Close()

	live, err := liveness.NewDetector(
		modelsDir+"/"+liveness.ModelV2FileName,
		modelsDir+"/"+liveness.ModelV1SEFileName,
	)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer live.Close()

	img, err := loadImage("../fotos/amy.jpg")
	if err != nil {
		fmt.Println(err)
		return
	}

	faces, err := det.Detect(img)
	if err != nil {
		fmt.Println(err)
		return
	}
	if len(faces) == 0 {
		fmt.Println("no face found")
		return
	}

	result, err := live.Detect(img, faces[0].Rectangle)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("IsLive=%v Score=%.4f\n", result.IsLive, result.Score)

}
