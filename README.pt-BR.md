## go-onnxface

[![CI](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml/badge.svg)](https://github.com/leandroveronezi/go-onnxface/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/leandroveronezi/go-onnxface.svg)](https://pkg.go.dev/github.com/leandroveronezi/go-onnxface)
![MIT Licensed](https://img.shields.io/github/license/leandroveronezi/go-onnxface.svg)

🇺🇸 [Read in English](README.md)

Detecção e reconhecimento facial para Go, baseado no [ONNX Runtime](https://onnxruntime.ai)
em vez de dlib. Projeto irmão do [go-face](https://github.com/leandroveronezi/go-face)/
[go-recognizer](https://github.com/leandroveronezi/go-recognizer), com foco em melhor
acurácia em condições reais (pose, iluminação) do que o modelo de 2017 do dlib, usando
múltiplos engines intercambiáveis -- cada um um subpacote implementando um contrato
compartilhado (`face.FaceDetector` ou `face.FaceRecognizer`), então adicionar um novo
não muda como os outros são usados.

Diferente do dlib, o ONNX Runtime não precisa ser compilado a partir do código-fonte: a
Microsoft publica bibliotecas compartilhadas pré-compiladas por plataforma.
`Recognizer.DownloadModels` baixa automaticamente a certa para seu SO/arquitetura, da
mesma forma que o `DownloadModels` do go-recognizer baixa os arquivos de modelo do dlib
-- sem passo manual de "caminho do .so".

### Engines

Cada modelo aqui foi escolhido por ter uma licença explícita e de uso comercial nos
*pesos publicados* -- não apenas código permissivo envolvendo pesos treinados em um
dataset de uso restrito a pesquisa (MS1M/CASIA-WebFace/VGGFace2 e similares), que é a
armadilha em que a maioria dos repositórios "licenciados como MIT" de reconhecimento
facial cai. Veja o link de cada linha para como isso foi verificado.

| Tipo | Pacote | Modelo | Licença | Notas |
|------|---------|-------|---------|-------|
| Detecção | `yunet` | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT | Padrão. Entrada fixa 640x640 (com letterbox). |
| Detecção | `centerface` | [CenterFace](https://github.com/Star-Clouds/CenterFace) | MIT | Tamanho de entrada dinâmico (redimensionado para múltiplo de 32, sem distorção de letterbox). |
| Detecção | `retinaface` | [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) (resnet50) | MIT | O mais pesado dos três (~52MB float16 vs ~7,5MB do CenterFace/~230KB do YuNet). Entrada fixa 640x640 (com letterbox). |
| Reconhecimento | `sface` | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache-2.0 | Único modelo de reconhecimento encontrado até agora com concessão comercial explícita sobre os pesos -- veja Licenciamento abaixo. |
| Reconhecimento | `arcface` | qualquer export ONNX da família ArcFace | *depende dos seus pesos* | Uma ponte, não um modelo: não embute nem baixa nenhum peso. Veja Licenciamento abaixo antes de usar. |
| Reconhecimento | `ghostface` | [GhostFaceNetV1](https://github.com/HamadYA/GhostFaceNets) | *depende dos seus pesos* | Mesma situação do `arcface` (treinado em MS1MV2/MS1MV3) -- uma ponte, sem pesos embutidos ou baixados. Moderno (2023) e competitivo com o ArcFace, diferente dos outros modelos de reconhecimento envolvidos pelo DeepFace (VGG-Face/OpenFace/"DeepFace" de 2014), que já são antigos demais para valer a pena adicionar. |
| Liveness | `liveness` | [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (MiniFASNetV2 + MiniFASNetV1SE) | Apache-2.0 | Engine de liveness padrão. Detecção de spoof por impressão/replay -- treinado especificamente para essa tarefa, não em um dataset de identidade facial, então nenhuma das ressalvas de licenciamento dos modelos de reconhecimento se aplica. |
| Liveness | `seetaface6` | [SeetaFace6](https://github.com/seetafaceengine/SeetaFace6) (fas_first + fas_second) | BSD | Pega spoof de impressão/replay de forma bem mais agressiva que o `liveness`, ao custo de rejeitar muito mais gente real de verdade -- uma troca real de precisão/recall, não uma melhoria estrita. Veja o README do go-onnxface-benchmarks pros números que validaram isso. |

Tanto `liveness` quanto `seetaface6` implementam o contrato compartilhado
`face.LivenessDetector`, então são intercambiáveis via
`Recognizer.Model.Liveness` (veja Uso abaixo) ou diretamente, igual aos
engines de detecção/reconhecimento.

Veja [Benchmarks](#benchmarks) abaixo pros números de acurácia/latência de cada pacote.

**Status**: em desenvolvimento inicial.
- ✅ Detecção (`yunet.Detector`, `centerface.Detector`, `retinaface.Detector`) --
  cada um validado contra sua própria execução de referência real em Python
  (caixa/landmarks/score batem com diferença de ~1-2px). Os arquivos `.onnx` do
  CenterFace e do RetinaFace não são downloads diretos do upstream (veja
  Desenvolvimento abaixo para o porquê e como foram produzidos) e, diferente de
  YuNet/SFace, também não são hospedados pelo próprio projeto upstream --
  `centerface.DownloadModel`/`retinaface.DownloadModel` baixam cópias próprias do
  go-onnxface (veja a seção Desenvolvimento).
- ✅ Reconhecimento (`sface.Recognizer`) -- `Align`/`Feature` validados contra uma
  execução real do `cv2.FaceRecognizerSF` (cosine ~1,0 mesma pessoa, ~0,11-0,12
  pessoas diferentes em ambas as implementações, bem abaixo do limiar de ~0,363
  do SFace para mesma pessoa).
- ✅ API fácil (`Recognizer`, orientada a caminho de arquivo, com download automático)
  e API de baixo nível (`Engine`, `Compare`, trabalhando com `image.Image`) -- veja
  Uso abaixo.
- ✅ Ponte `arcface` (só código, traga seus próprios pesos) -- validada localmente
  contra uma execução real do `buffalo_l` (`w600k_r50.onnx`) da InsightFace (cosine
  ~1,0 mesma pessoa, ~-0,03 pessoas diferentes); não faz parte do CI porque precisa
  de pesos que este repositório não pode distribuir.
- ✅ Liveness/anti-spoof (`liveness.Detector`) -- a matemática de recorte/pré-
  processamento/ensemble portada linha a linha da própria fonte do
  Silent-Face-Anti-Spoofing (não só do README), validada contra uma execução real
  via onnxruntime dos modelos originais sem modificação em `testdata/amy.jpg` (uma
  foto real, corretamente classificada como ao vivo, ~0,99).
- ✅ Liveness/anti-spoof (`seetaface6.Detector`) -- a fusão fas_first/fas_second
  (gate SSD de imagem inteira, classificador de recorte alinhado convertido pra
  YCrCb) portada linha a linha do código-fonte C++ do próprio SeetaFace6
  (`FaceAntiSpoofingX/src/seeta/FaceAntiSpoofing.cpp`), validada contra uma
  execução real via onnxruntime em `testdata/amy.jpg` (corretamente classificada
  como ao vivo, ~0,999).
- ✅ Ponte `ghostface` (só código, traga seus próprios pesos, mesmo raciocínio do
  `arcface`) -- validada localmente contra uma execução real do GhostFaceNetV1,
  convertido dos pesos originais em Keras via tf2onnx (cosine ~1,0 mesma pessoa,
  ~0,008 pessoas diferentes, usando exatamente os landmarks que o próprio lado Go
  produziu -- confirma que o port do alinhamento/pré-processamento está correto,
  não só "próximo o suficiente").

### Benchmarks

Medido contra datasets reais e não curados -- escolhidos especificamente por
incluírem variação real de pose/iluminação/oclusão, em vez de fotos de estúdio
posadas:

- **Detecção**: conjunto de validação do [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
  -- 3.226 imagens, 19.926 rostos de referência com ≥20px de altura. Métrica: recall
  com IoU≥0,5, todos os níveis de dificuldade combinados. Isso *não* é o protocolo
  oficial do eval-tools do WIDER FACE (subconjuntos separados de fácil/médio/difícil,
  average precision estilo PASCAL-VOC), então trate as duas colunas abaixo como
  comparáveis em direção, não como a mesma métrica.
- **Reconhecimento**: [CFP-FP](http://www.cfpw.io/) (Celebrities in Frontal-Profile)
  -- 7.000 pares de verificação frontal-vs-perfil em todos os 10 folds -- e
  [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/) -- 7.000 pares gerados
  localmente (identidade igual/diferente, seed fixa) a partir de um reempacotamento
  com 567 identidades/16.488 imagens, já que o AgeDB não vem com um protocolo fixo
  como o CFP-FP. Métrica pros dois: acurácia no único limiar de melhor desempenho
  possível, varrido post-hoc sobre todos os pares -- um pouco mais otimista que o
  protocolo padrão de limiar validado por 10 folds cruzados que a maioria dos papers
  reporta.
- **Liveness**: split de teste do [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)
  -- ~6.700 imagens (1 dos 10 shards públicos), proporção aproximada de 30%/70%
  vivo/spoof.

**Detecção (WIDER FACE):**

| Pacote | Recall | Latência média/imagem (CPU, veja nota) | Referência publicada (modelo/paper original) |
|---------|--------|-------------------------------------------|----------------------------------------------|
| `yunet` | 70,67% | 36,8ms | 88,44% / 86,56% / 75,03% AP fácil/médio/difícil ([opencv_zoo](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)) |
| `centerface` | 78,92% | 247,5ms | 92,2% / 91,1% / 78,2% mAP fácil/médio/difícil, escala única ([upstream](https://github.com/Star-Clouds/CenterFace)) |
| `retinaface` | 76,55% | 384,9ms | 96,5% / 95,6% / 90,4% mAP fácil/médio/difícil ([paper](https://arxiv.org/abs/1905.00641)) |

**Reconhecimento (CFP-FP e AgeDB, uma coluna pra cada):**

| Pacote | Acurácia CFP-FP | Acurácia AgeDB | Latência média/imagem (CPU) | Referência publicada |
|---------|------------------|-----------------|---------------------------|----------------------|
| `sface` | 97,13% | 95,47% | 15,5ms | 95,26% CFP-FP ([paper](https://arxiv.org/abs/2205.12010), configuração ResNet50/CASIA-WebFace -- os pesos distribuídos são um MobileFaceNet mais leve, não necessariamente idêntico) |
| `arcface` (buffalo_l) | 99,51% | 97,78% | 112,9ms | 99,33% CFP-FP ([model zoo da InsightFace](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)) |
| `ghostface` | 96,80% | 96,48% | 13,7ms | 96,83% CFP-FP ([paper](https://www.researchgate.net/publication/369930264_GhostFaceNets_Lightweight_Face_Recognition_Model_from_Cheap_Operations)) |

**Liveness (CelebA-Spoof):**

| Pacote | Acurácia vivo | Acurácia spoof | Latência média/imagem (CPU) |
|---------|----------------|------------------|---------------------------|
| `liveness` | 74,18% | 69,67% | ~12ms |

Latência é inferência só em CPU (sem GPU), medida em um Intel i7-1165G7
(4 núcleos/8 threads) -- trate como direcional pro seu próprio hardware,
não como número absoluto. O AgeDB é um benchmark mais difícil que o CFP-FP
pra todo mundo (testa variação de idade, não só pose frontal-vs-perfil) --
o `arcface` cai mais que os outros (99,51%->97,78%), sugerindo que ele
generaliza um pouco pior pra variação de idade especificamente. Os
números do `liveness` no CelebA-Spoof são bem mais baixos do que uma foto
real de chamada em sala de aula produziria (veja a validação específica
de print/replay já descrita acima) -- as imagens do CelebA-Spoof são
capturas de câmera de vigilância em baixa resolução, que exigem mais do
modelo do que uma foto de celular/webcam.

Pose é consistentemente o atributo mais difícil para os três detectores: os rostos com
pose atípica do WIDER FACE derrubam o recall de ~73-81% (pose típica) para ~38-58%
(atípica) -- uma queda maior do que blur ou oclusão causam sozinhos. O RetinaFace é o
mais robusto especificamente a pose atípica (57,68% vs 53,15% do CenterFace e 38,37%
do YuNet); o CenterFace tem o melhor recall geral e o melhor tratamento de
oclusão/iluminação. Nenhum dos três -- ou qualquer detector facial -- consegue achar
um rosto que não está visível de forma alguma (ex: alguém olhando reto para baixo):
isso é um problema de detecção de pessoa/cabeça, não de detecção facial, e está fora
do escopo aqui.

### Licenciamento: por que tão poucos modelos de reconhecimento

Modelos de *detecção* facial (YuNet, CenterFace, RetinaFace, MTCNN, ...) treinam no
WIDER FACE -- só caixas delimitadoras, sem rótulos de identidade -- então licenciá-los
comercialmente costuma ser tranquilo. Modelos de *reconhecimento* facial precisam de
dados rotulados por identidade, e quase todo modelo conhecido (ArcFace, GhostFaceNets,
FaceNet, VGG-Face, Buffalo_L, ...) é treinado em datasets da linhagem
MS1M/CASIA-WebFace/VGGFace2 liberados "apenas para fins de pesquisa" -- essa restrição
se aplica aos pesos independentemente da licença do *código* (código MIT em torno de
pesos não comerciais continua sendo não comercial na prática). SFace é a única exceção
encontrada até agora: a OpenCV Zoo distribui esses pesos específicos sob uma concessão
explícita Apache-2.0.

Os pacotes `arcface`/`ghostface` existem para modelos dessa família mesmo assim, mas
deliberadamente como *só código*: nenhum dos dois embute ou baixa nenhum arquivo
`.onnx`, porque isso significaria este repositório licenciado como MIT redistribuindo
pesos não comerciais de terceiros -- a licença da ponte não muda a licença que os
pesos carregam. Traga seu próprio arquivo, e garanta que você tem os direitos para o
uso que pretende dar: uso de pesquisa dos pesos `buffalo_l`/`antelopev2` publicados
pela InsightFace geralmente é permitido pelos termos deles, uso comercial precisa da
licença paga deles (veja a página de licenciamento comercial deles). Os pesos do
GhostFaceNets, treinados em MS1MV2/MS1MV3, carregam a mesma restrição de uso apenas
para pesquisa, mas -- diferente da InsightFace -- o HamadYA não oferece nenhuma
licença comercial paga, então uso comercial desse arquivo de pesos específico não é
algo que você consegue comprar hoje; um retreino do zero com seus próprios dados
licenciados seria a única rota. Isso não é aconselhamento jurídico, é só como o
licenciamento do ecossistema funciona na prática -- na dúvida, pergunte a um
advogado, não a este README.

## Uso

O jeito fácil -- espelha o go-recognizer: aponte para um diretório, tudo mais
(biblioteca compartilhada do onnxruntime, modelos de detecção/reconhecimento) é
baixado automaticamente para lá e só é baixado de novo se estiver faltando:

```go
import "github.com/leandroveronezi/go-onnxface"

rec := &onnxface.Recognizer{}

if err := rec.DownloadModels("models"); err != nil {
    // ...
}
if err := rec.Init("models"); err != nil {
    // ...
}
defer rec.Close()

rec.AddImageToDataset("amy.jpg", "Amy")

result, err := rec.Identify("photo.jpg")
switch {
case errors.Is(err, onnxface.ErrNoFace), errors.Is(err, onnxface.ErrMultipleFaces):
    // photo.jpg não tem exatamente um rosto
case errors.Is(err, onnxface.ErrNoMatch):
    // nenhuma entrada do Dataset dentro de rec.Tolerance
case err != nil:
    // outro erro qualquer (I/O, decodificação, ...)
}
fmt.Println(result.Id, result.Distance, result.Confidence)
```

`AddImageToDataset`/`RecognizeSingle`/`Identify` retornam esses erros
sentinela para as condições de falha "esperadas" -- verifique com
`errors.Is` em vez de comparar o texto da mensagem de erro, que não faz
parte do contrato da API e pode mudar. Qualquer outro erro (I/O,
decodificação de imagem, etc.) vem envolvido com `%w`, então
`errors.Unwrap`/`errors.As` ainda alcançam a causa original.

Por padrão o `Recognizer` usa `yunet`+`sface` e nenhum engine de liveness
-- configure `Model` antes de `Init` pra escolher outros (qualquer
combinação da tabela de Engines acima), mais os overrides de
arquivo/config específicos de cada engine:

```go
rec := &onnxface.Recognizer{}
rec.Model.Detector = onnxface.DetectorRetinaFace  // mais preciso, mais lento -- veja Benchmarks
rec.Model.Liveness = onnxface.LivenessSeetaFace6  // adiciona o CheckLiveness abaixo

if err := rec.DownloadModels("models"); err != nil { // baixa só o que Model seleciona
    // ...
}
if err := rec.Init("models"); err != nil {
    // ...
}
defer rec.Close()

liveness, err := rec.CheckLiveness("photo.jpg")
switch {
case errors.Is(err, onnxface.ErrNoLivenessEngine):
    // Model.Liveness ficou no padrão (LivenessNone)
case err != nil:
    // photo.jpg não tem exatamente um rosto, ou outro erro qualquer
default:
    fmt.Println(liveness.IsLive, liveness.Score)
}
```

`RecognizerArcFace`/`RecognizerGhostFace` não têm pesos embutidos (veja
Licenciamento acima), então `Model.RecognizerFile` e, no caso do
`arcface`, `Model.ArcFace` são obrigatórios -- `Init` retorna um erro
claro se faltar, e `DownloadModels` vira no-op pros dois.

`Recognizer.Tolerance` tem padrão 1,128 (limiar L2 sugerido
pela OpenCV para o SFace) depois de `Init`; ajuste para o seu próprio uso, mesma ideia
do `Tolerance` do go-face/go-recognizer. `SaveDataset`/`LoadDataset` persistem o
`Dataset` de/para um arquivo JSON.

Controle de baixo nível -- trabalhe com `image.Image` diretamente, escolha seu
próprio detector/reconhecedor (ex: `centerface` em vez de `yunet`), ou rode
detecção/reconhecimento como passos separados:

```go
import (
    "github.com/leandroveronezi/go-onnxface"
    "github.com/leandroveronezi/go-onnxface/centerface"
    "github.com/leandroveronezi/go-onnxface/sface"
)

onnxface.InitEnvironment("/path/to/libonnxruntime.so")
defer onnxface.CloseEnvironment()

// Cada engine baixa só o próprio modelo -- usar centerface aqui nunca
// baixa yunet/sface também, e vice-versa.
centerface.DownloadModel("models")

det, _ := centerface.NewDetector("models/centerface.onnx")
rec, _ := sface.NewRecognizer("models/face_recognition_sface_2021dec.onnx")

engine := onnxface.NewEngine(det, rec)
defer engine.Close()

results, _ := engine.Recognize(img) // img é um image.Image padrão
for _, r := range results {
    fmt.Println(r.Rectangle, r.Landmarks, r.Score, len(r.Feature))
}

result := onnxface.Compare(results[0].Feature, knownFeature, 1.128)
fmt.Println(result.IsMatch, result.Distance, result.Confidence)
```

`yunet`/`centerface`/`sface`/`arcface` implementam o contrato compartilhado
`onnxface.FaceDetector`/`FaceRecognizer` (no subpacote `face`) -- um futuro engine é
um novo subpacote implementando esse contrato, não uma mudança no código existente.
Veja `examples/` para programas completos e executáveis.

Usar `arcface` (veja a nota de Licenciamento acima) significa fornecer seu próprio
arquivo de modelo mais os nomes dos tensores, já que eles variam entre exports do
ArcFace:

```go
rec, _ := arcface.NewRecognizer("/path/to/your/model.onnx", arcface.Config{
    InputName:  "input.1", // inspecione seu próprio arquivo -- veja a doc do pacote arcface
    OutputName: "683",
})
```

`ghostface` é uma ponte mais estreita (uma arquitetura específica, GhostFaceNetV1,
não "qualquer export ONNX dessa família" como o `arcface`), então não precisa de
`Config` -- basta apontar para sua própria conversão (veja a doc do pacote para a
receita exata):

```go
rec, _ := ghostface.NewRecognizer("/path/to/your/ghostfacenet.onnx")
```

Detecção de liveness recebe um retângulo de rosto de qualquer detector (não está
preso ao `face.FaceDetector`) e classifica como rosto ao vivo ou spoof por
impressão/replay:

```go
import "github.com/leandroveronezi/go-onnxface/liveness"

liveness.DownloadModel("models") // os dois modelos do ensemble juntos -- sempre usados em par

live, _ := liveness.NewDetector("models/minifasnet_v2.onnx", "models/minifasnet_v1se.onnx")
defer live.Close()

faces, _ := det.Detect(img) // qualquer FaceDetector
result, _ := live.Detect(img, faces[0].Rectangle)
fmt.Println(result.IsLive, result.Score)
```

## Requisitos

- Go com suporte a cgo.
- A biblioteca compartilhada do onnxruntime (versão 1.26.0) para sua plataforma.
  `Recognizer.DownloadModels` a baixa automaticamente para linux/amd64, linux/arm64,
  darwin/arm64, windows/amd64 e windows/arm64 (verificado contra os
  [assets reais da release v1.26.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0);
  darwin/amd64 não tem pré-compilado para essa versão). Para qualquer outra
  plataforma, baixe você mesmo e aponte `InitEnvironment` diretamente para ela.

## Desenvolvimento

Nenhum peso de modelo é versionado neste repositório (veja Licenciamento acima para
o porquê, pelo menos no caso do `arcface` -- para os outros é para manter os clones
pequenos e evitar os limites de tamanho de arquivo do GitHub, o mesmo motivo pelo
qual os próprios pesos do DeepFace ficam em uma release separada em vez do
repositório principal). Baixe o que os testes precisam para `models/` antes de
rodá-los:

```bash
curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz
tar xzf ort.tgz
export ONNXFACE_ORT_LIB="$PWD/onnxruntime-linux-x64-1.26.0/lib/libonnxruntime.so.1.26.0"

mkdir -p models
curl -sL -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
curl -sL -o models/centerface.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/centerface.onnx
curl -sL -o models/retinaface.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/retinaface.onnx
curl -sL -o models/minifasnet_v2.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/minifasnet_v2.onnx
curl -sL -o models/minifasnet_v1se.onnx https://github.com/leandroveronezi/go-onnxface/releases/download/models-v1/minifasnet_v1se.onnx

go test ./...
```

Testes que precisam da biblioteca compartilhada do onnxruntime ou de um arquivo de
modelo pulam a si mesmos se não estiverem presentes, então um download parcial só
pula menos testes em vez de falhar os que não precisam do que está faltando.
`TestDownloadModels`/`centerface.TestDownloadModel`/`retinaface.TestDownloadModel`/
`liveness.TestDownloadModel` adicionalmente fazem um download de rede real para um
diretório novo cada um, para verificar que o próprio caminho de download (não só um
arquivo pré-baixado) funciona de ponta a ponta. O teste do `arcface` precisa do seu
próprio modelo fornecido localmente (com a licença correta) e pula a si mesmo sem
ele -- veja o comentário de doc em `TestRecognizerAgainstLocalModel` em
`arcface/recognizer_test.go`.

### Por que centerface.onnx/retinaface.onnx/minifasnet_*.onnx não são downloads diretos do upstream

- **centerface.onnx**: o próprio arquivo publicado pelo Star-Clouds declara uma
  entrada fixa `[10,3,32,32]` (um artefato de trace do PyTorch sem `dynamic_axes`
  definido) que o ONNX Runtime aplica estritamente, mesmo que o `cv2.dnn` -- o que a
  própria implementação de referência do CenterFace usa -- tolere isso. Esta cópia
  só teve seus metadados de shape relaxados para batch/altura/largura dinâmicos
  (via `onnx.save`, sem tocar nos pesos treinados) para funcionar com o
  `DynamicAdvancedSession` do `onnxruntime_go`.
- **retinaface.onnx**: InsightFace/biubug6 só publicam pesos `.pth` do PyTorch,
  nenhum export ONNX. Produzido com o próprio `convert_to_onnx.py` do repositório
  upstream contra o `Resnet50_Final.pth` publicado por eles (chaves do state_dict
  batendo sem nada faltando ou inesperado, confirmando o checkpoint certo), entrada
  fixa 640x640. O export fp32 tinha ~109MB, acima do limite de 100MB do git do
  GitHub (assets de release não têm esse limite, mas o arquivo já estava sendo
  produzido de qualquer forma); os pesos (não os tensores float32 de entrada/saída)
  foram convertidos para float16 para chegar a ~52MB, revalidado depois (diferenças
  de caixa/landmark em relação à versão fp32 são de ~0,01px).
- **minifasnet_v2.onnx/minifasnet_v1se.onnx**: mesma história do retinaface.onnx --
  a minivision-ai só publica pesos `.pth` do PyTorch. Produzidos com as próprias
  definições de modelo do repositório upstream (`src/model_lib/MiniFASNet.py`)
  contra os `2.7_80x80_MiniFASNetV2.pth`/`4_0_0_80x80_MiniFASNetV1SE.pth`
  publicados por eles, entrada fixa 80x80, pequenos o suficiente (~1,7MB cada) para
  não precisar de conversão para float16.

Todos são MIT/Apache-2.0 (iguais aos projetos upstream) e treinados em tarefas sem
dados rotulados por identidade (caixas delimitadoras do WIDER FACE, ou os próprios
dados de ao-vivo/spoof do Silent-Face-Anti-Spoofing), então hospedar cópias
modificadas/reexportadas na própria
[release `models-v1`](https://github.com/leandroveronezi/go-onnxface/releases/tag/models-v1)
do go-onnxface não muda nada sobre a licença delas.

## Licença

[MIT](LICENSE)
