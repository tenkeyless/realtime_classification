import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:realtime_classification/classification_util.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as imglib;

class MobileNetV2RTClassificationScreen extends StatefulWidget {
  const MobileNetV2RTClassificationScreen({
    super.key,
  });

  @override
  State<MobileNetV2RTClassificationScreen> createState() =>
      _MobileNetV2RTClassificationScreenState();
}

void runIsolate(List<dynamic> message) async {
  SendPort sendPort = message[0];
  CameraImage image = message[1];
  List<String> labels = message[2];
  int interpreterAddr = message[3];

  final interpreter = Interpreter.fromAddress(interpreterAddr);
  final height = interpreter.getInputTensor(0).shape[1];

  final classificationResult = await processImage(
    image,
    labels,
    interpreter,
    height,
  );
  sendPort.send(classificationResult);
}

Future<Map<String, num>?> processImage(
  CameraImage imageInput,
  List<String> labels,
  Interpreter interpreter,
  int imageSize,
) async {
  final img = await convertImagetoPng(imageInput);

  if (img != null) {
    final resizedImg = imglib.copyResizeCropSquare(img, size: imageSize);
    final imageList = imgToList(resizedImg);
    final preprocessedImageList = imageList
        .map((e) =>
            e.map((e2) => e2.map((e3) => e3 / 127.5 - 1).toList()).toList())
        .toList();
    return imgClassification(
      preprocessedImageList,
      labels,
      interpreter,
    );
  }
  return null;
}

class _MobileNetV2RTClassificationScreenState
    extends State<MobileNetV2RTClassificationScreen> {
  static const _modelPath = 'assets/mobilenet_v2_model_1_f16.tflite';
  static const _labelsPath = 'assets/imagenet_label.txt';

  /// Labels
  late final List<String> labels;

  /// TF Interpreter
  late final Interpreter interpreter;

  late List<CameraDescription> _cameras;
  late CameraController _cameraController;

  Future<void> _startCamera() async {
    _cameras = await availableCameras();
    _cameraController = CameraController(
      _cameras[0],
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup:
          Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
    );
  }

  late Future<void> _isReadyToRealtimeInference;

  final ValueNotifier<Map<String, num>?> _classification = ValueNotifier(null);

  Future<void> _loadModel() async {
    interpreter = await getModel(_modelPath);
  }

  Future<void> _loadLabels() async {
    final labelTxt = await rootBundle.loadString(_labelsPath);
    labels = labelTxt.split('\n');
    labels.removeLast();
  }

  bool _isProcessing = false;

  Future<void> initCM() async {
    await _cameraController.startImageStream(
      (image) async {
        if (!_isProcessing) {
          _isProcessing = true;

          ReceivePort port = ReceivePort();
          final isolate = await Isolate.spawn<List<dynamic>>(
            runIsolate,
            [port.sendPort, image, labels, interpreter.address],
          );
          _classification.value = await port.first;
          isolate.kill(priority: Isolate.immediate);

          _isProcessing = false;
        }
      },
    );
  }

  @override
  void initState() {
    super.initState();

    // Camera and inference
    _isReadyToRealtimeInference =
        _startCamera().then((_) => _cameraController.initialize()).then(
      (_) {
        _loadLabels();
        _loadModel();
      },
    ).then(
      (_) {
        initCM();
        setState(() {});
      },
    );
  }

  @override
  void dispose() {
    _cameraController.stopImageStream();
    // interpreter.close();
    _cameraController.dispose();

    super.dispose();
  }

  DateTime lastUpdated = DateTime.now();
  double sum = 0.0;
  double cnt = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("MobileNet V2 Classification"),
      ),
      body: Stack(
        children: [
          FutureBuilder(
            future: _isReadyToRealtimeInference,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return AspectRatio(
                  aspectRatio: 1 / 1,
                  child: ClipRect(
                    child: Transform.scale(
                      scale: _cameraController.value.aspectRatio / 1,
                      child: Center(
                        child: CameraPreview(_cameraController),
                      ),
                    ),
                  ),
                );
              } else {
                return const Center(child: CircularProgressIndicator());
              }
            },
          ),
          Column(
            children: [
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 20,
                ),
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.3),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      "분류 결과",
                      style: TextStyle(
                        fontSize: 18,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 8),
                    ValueListenableBuilder(
                      valueListenable: _classification,
                      builder: (context, value, child) {
                        final now = DateTime.now();
                        final difference = now
                                .difference(lastUpdated)
                                .inMilliseconds
                                .toDouble() /
                            1000;
                        lastUpdated = now;
                        cnt++;
                        sum += difference;
                        final avg = sum / cnt;
                        final fps = 1 ~/ avg;
                        if (value != null) {
                          return Expanded(
                            child: SingleChildScrollView(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                      "이번 시간: ${difference.toStringAsFixed(3)}"),
                                  Text("누적 횟수: ${cnt.toInt()}"),
                                  Text(
                                      "누적 평균: ${avg.toStringAsFixed(3)} - 평균 FPS: $fps"),
                                  ...(value.entries.toList()
                                        ..sort(
                                          (a, b) => a.value.compareTo(b.value),
                                        ))
                                      .reversed
                                      .take(4)
                                      .map(
                                        (e) => Text(
                                          "${e.key}: ${e.value}",
                                          style: const TextStyle(
                                            fontSize: 14,
                                            color: Colors.white,
                                          ),
                                        ),
                                      ),
                                ],
                              ),
                            ),
                          );
                        } else {
                          return const Text('Loading');
                        }
                      },
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
