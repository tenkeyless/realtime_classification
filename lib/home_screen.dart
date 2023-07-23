import 'package:flutter/material.dart';
import 'package:realtime_classification/load_tflite.dart';
import 'package:realtime_classification/mobilenet_v2_r_classification_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("DL")),
      body: Align(
        alignment: Alignment.topCenter,
        child: SingleChildScrollView(
          child: Column(children: [
            homeListButton(
              context,
              const LoadTFLiteScreen(),
              "Load tflite",
            ),
            homeListButton(
              context,
              const MobileNetV2RTClassificationScreen(),
              "Classification",
            ),
          ]),
        ),
      ),
    );
  }

  ElevatedButton homeListButton(
    BuildContext context,
    Widget moveTo,
    String title,
  ) {
    return ElevatedButton(
      onPressed: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => moveTo,
          ),
        );
      },
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 16,
        ),
      ),
    );
  }
}
