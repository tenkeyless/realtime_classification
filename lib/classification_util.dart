import 'dart:io';

import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as imglib;

Future<Interpreter> getModel(String modelPath, {int? cpuThreadNum}) async {
  final options = InterpreterOptions();

  if (cpuThreadNum != null) {
    options.threads = cpuThreadNum;
  }

  // XNNPACK 델리게이트 사용
  if (Platform.isAndroid) {
    options.addDelegate(XNNPackDelegate());
  }

  // GPU 델리게이트 사용
  // 에뮬레이터에서는 동작하지 않음
  // if (Platform.isAndroid) {
  //   options.addDelegate(GpuDelegateV2());
  // }

  // Metal 델리게이트 사용
  if (Platform.isIOS) {
    options.addDelegate(GpuDelegate());

    // final delegateOpts = GpuDelegateOptions(waitType: 1);
    // options.addDelegate(GpuDelegate(options: delegateOpts));
  }

  // assets에서 모델 로드
  return await Interpreter.fromAsset(
    modelPath,
    options: options,
  );
}

imglib.Image getImage(String imagePath, int imageWidth, int imageHeight) {
  // 파일에서 이미지 바이트로 읽기
  final imageData = File(imagePath).readAsBytesSync();

  // package:image/image.dart (https://pub.dev/image)를 사용한 이미지 디코딩
  final image = imglib.decodeImage(imageData);

  // 모델 입력을 위한 이미지 리사이즈
  return imglib.copyResize(
    image!,
    width: imageWidth,
    height: imageHeight,
  );
}

List<List<List<num>>> imgToList(imglib.Image imageInput) {
  // 이미지 행렬로 전환
  return List.generate(
    imageInput.height,
    (y) => List.generate(
      imageInput.width,
      (x) {
        final pixel = imageInput.getPixel(x, y);
        return [pixel.r, pixel.g, pixel.b];
      },
    ),
  );
}

Map<String, num> imgClassification(
  List<List<List<num>>> imageMatrix,
  List<String> labels,
  Interpreter interpreter,
) {
  // 입력 텐서 가져오기 [1, 28, 28, 3]
  final input = [imageMatrix];
  // 출력 텐서 세팅 [1, 5]
  final labelNum = labels.length;
  final output = [List<num>.filled(labelNum, 0)];

  // 추론 실행
  // final preStartedAt = DateTime.now();
  interpreter.run(input, output);
  // print("run time: ${DateTime.now().difference(preStartedAt).inMilliseconds}");

  // 첫 번째 출력 가져오기
  final result = output.first;

  // 분류 맵 {label: points} 설정
  final classification = <String, num>{};

  for (var i = 0; i < result.length; i++) {
    if (result[i] != 0) {
      // 라벨:포인트 설정
      classification[labels[i]] = result[i];
    }
  }

  return classification;
}

Future<imglib.Image?> convertImagetoPng(CameraImage image) async {
  try {
    imglib.Image? img;
    if (image.format.group == ImageFormatGroup.yuv420) {
      // img = _convertYUV420(image);
      img = _convertYUV420toImageColor(image);
      // img = _convertYUV420ToImage(image);
    } else if (image.format.group == ImageFormatGroup.bgra8888) {
      img = _convertBGRA8888(image);
    }

    return img;
  } catch (e) {
    print(">>>>>>>>>>>> ERROR:$e");
  }
  return null;
}

// CameraImage BGRA8888 -> PNG
// Color
imglib.Image _convertBGRA8888(CameraImage image) {
  return imglib.Image.fromBytes(
    width: image.width,
    height: image.height,
    bytes: image.planes[0].bytes.buffer,
    format: imglib.Format.uint8,
    order: imglib.ChannelOrder.rgba,
    // format: imglib.Format.bgra,
  );
}

imglib.Image _convertYUV420toImageColor(CameraImage image) {
  const shift = (0xFF << 24);

  final int width = image.width;
  final int height = image.height;
  final int uvRowStride = image.planes[1].bytesPerRow;
  final int uvPixelStride = image.planes[1].bytesPerPixel!;

  final img = imglib.Image(height: height, width: width); // Create Image buffer

  // Fill image buffer with plane[0] from YUV420_888
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      final int uvIndex =
          uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
      final int index = y * width + x;

      final yp = image.planes[0].bytes[index];
      final up = image.planes[1].bytes[uvIndex];
      final vp = image.planes[2].bytes[uvIndex];
      // Calculate pixel color
      int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
      int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
          .round()
          .clamp(0, 255);
      int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
      // color: 0x FF  FF  FF  FF
      //           A   B   G   R
      if (img.isBoundsSafe(height - y, x)) {
        img.setPixelRgba(height - y, x, r, g, b, shift);
      }
    }
  }

  return img;
}

imglib.Image _convertYUV420ToImage(CameraImage cameraImage) {
  final imageWidth = cameraImage.width;
  final imageHeight = cameraImage.height;

  final yBuffer = cameraImage.planes[0].bytes;
  final uBuffer = cameraImage.planes[1].bytes;
  final vBuffer = cameraImage.planes[2].bytes;

  final int yRowStride = cameraImage.planes[0].bytesPerRow;
  final int yPixelStride = cameraImage.planes[0].bytesPerPixel!;

  final int uvRowStride = cameraImage.planes[1].bytesPerRow;
  final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

  final image = imglib.Image(width: imageWidth, height: imageHeight);

  for (int h = 0; h < imageHeight; h++) {
    int uvh = (h / 2).floor();

    for (int w = 0; w < imageWidth; w++) {
      int uvw = (w / 2).floor();

      final yIndex = (h * yRowStride) + (w * yPixelStride);

      // Y plane should have positive values belonging to [0...255]
      final int y = yBuffer[yIndex];

      // U/V Values are subsampled i.e. each pixel in U/V chanel in a
      // YUV_420 image act as chroma value for 4 neighbouring pixels
      final int uvIndex = (uvh * uvRowStride) + (uvw * uvPixelStride);

      // U/V values ideally fall under [-0.5, 0.5] range. To fit them into
      // [0, 255] range they are scaled up and centered to 128.
      // Operation below brings U/V values to [-128, 127].
      final int u = uBuffer[uvIndex];
      final int v = vBuffer[uvIndex];

      // Compute RGB values per formula above.
      int r = (y + v * 1436 / 1024 - 179).round();
      int g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
      int b = (y + u * 1814 / 1024 - 227).round();

      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      image.setPixelRgb(w, h, r, g, b);
    }
  }

  return image;
}

imglib.Image _convertYUV420ToImage2(CameraImage cameraImage) {
  final imageWidth = cameraImage.width;
  final imageHeight = cameraImage.height;

  final yBuffer = cameraImage.planes[0].bytes;
  final uBuffer = cameraImage.planes[1].bytes;
  final vBuffer = cameraImage.planes[2].bytes;

  final int yRowStride = cameraImage.planes[0].bytesPerRow;
  final int yPixelStride = cameraImage.planes[0].bytesPerPixel!;

  final int uvRowStride = cameraImage.planes[1].bytesPerRow;
  final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

  final image = imglib.Image(width: imageWidth, height: imageHeight);

  for (int h = 0; h < imageHeight; h++) {
    int uvh = (h / 2).floor();

    for (int w = 0; w < imageWidth; w++) {
      int uvw = (w / 2).floor();

      final yIndex = (h * yRowStride) + (w * yPixelStride);

      // Y plane should have positive values belonging to [0...255]
      final int y = yBuffer[yIndex];

      // U/V Values are subsampled i.e. each pixel in U/V chanel in a
      // YUV_420 image act as chroma value for 4 neighbouring pixels
      final int uvIndex = (uvh * uvRowStride) + (uvw * uvPixelStride);

      // U/V values ideally fall under [-0.5, 0.5] range. To fit them into
      // [0, 255] range they are scaled up and centered to 128.
      // Operation below brings U/V values to [-128, 127].
      final int u = uBuffer[uvIndex];
      final int v = vBuffer[uvIndex];

      // Compute RGB values per formula above.
      int r = (y + v * 1436 / 1024 - 179).round();
      int g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
      int b = (y + u * 1814 / 1024 - 227).round();

      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      image.setPixelRgb(w, h, r, g, b);
    }
  }
  return image;
}
