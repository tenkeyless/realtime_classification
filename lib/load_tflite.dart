import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class LoadTFLiteScreen extends StatefulWidget {
  const LoadTFLiteScreen({super.key});

  @override
  State<LoadTFLiteScreen> createState() => _LoadTFLiteScreenState();
}

Interpreter? interpreter;

class _LoadTFLiteScreenState extends State<LoadTFLiteScreen> {
  final _modelPath = 'assets/mobilenet_v2_model_1_f16.tflite';
  final _labelsPath = 'assets/imagenet_label.txt';

  // late final Interpreter interpreter;
  late final List<String> labels;

  Future<Interpreter> getModel(String modelPath, {int? cpuThreadNum}) async {
    final options = InterpreterOptions();

    if (cpuThreadNum != null) {
      options.threads = cpuThreadNum;
    }

    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate());
    }

    if (Platform.isIOS) {
      options.addDelegate(GpuDelegate());
    }

    return await Interpreter.fromAsset(modelPath, options: options);
  }

  Future<void> loadModel() async {
    interpreter ??= await getModel(_modelPath);
    setState(() {});
  }

  Future<void> loadLabels() async {
    final labelTxt = await rootBundle.loadString(_labelsPath);
    labels = labelTxt.split('\n');
    labels.removeLast();
  }

  @override
  void initState() {
    super.initState();

    loadModel();
    loadLabels();
  }

  @override
  void dispose() {
    final interpreterDeleted = interpreter?.isDeleted;
    if (interpreterDeleted != null && !interpreterDeleted) {
      interpreter?.close();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Load TFLite..")),
    );
  }
}
