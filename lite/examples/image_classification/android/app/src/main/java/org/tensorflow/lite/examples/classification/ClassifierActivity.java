/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Model;


public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  private Bitmap rgbFrameBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier classifier;
  private BorderedText borderedText;
  /** Input image size of the model along x axis. */
  private int imageSizeX;
  /** Input image size of the model along y axis. */
  private int imageSizeY;

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_ic_camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    recreateClassifier(getModel(), getDevice(), getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
  }

  private static String MD5_Hash(String s) {
    MessageDigest m = null;

    try {
      m = MessageDigest.getInstance("MD5");
    } catch (NoSuchAlgorithmException e) {
      e.printStackTrace();
    }

    m.update(s.getBytes(),0,s.length());
    String hash = new BigInteger(1, m.digest()).toString(16);
    return hash;
  }

  @Override
  protected void processImage() {
    //////////////////////////////////////////////////////////////////////////////////////////
    //rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    LOGGER.i("Android Version: %s", android.os.Build.VERSION.RELEASE);
        runInBackground(
                new Runnable() {
                  @Override
                  public void run() {
                    AssetManager assetManager = getAssets();
                    InputStream assetInStream=null;
                    try {
                      String[] images = assetManager.list("test_images");
                      for(String imgName : images) {
                        assetInStream = getAssets().open("test_images/" + imgName);
                        rgbFrameBitmap = BitmapFactory.decodeStream(assetInStream);
                        final int cropSize = Math.min(previewWidth, previewHeight);
                        if (classifier != null) {
                          final long startTime = SystemClock.uptimeMillis();
                          final List<Classifier.Recognition> results =
                                  classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
                          lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                          ByteArrayOutputStream stream = new ByteArrayOutputStream();
                          rgbFrameBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                          byte[] byteArray = stream.toByteArray();
                          String md5 = MD5_Hash(new String(byteArray));
                          LOGGER.i("Image Name: %s", imgName);
                          LOGGER.i("MD5 Hash: %s", md5);
                          LOGGER.i("Detect: %s", results);
                          LOGGER.i("Time: %d ms", lastProcessingTimeMs);

                          runOnUiThread(
                                  new Runnable() {
                                    @Override
                                    public void run() {
                                      showResultsInBottomSheet(results);
                                      showFrameInfo(previewWidth + "x" + previewHeight);
                                      showCropInfo(imageSizeX + "x" + imageSizeY);
                                      showCameraResolution(cropSize + "x" + cropSize);
                                      showRotationInfo(String.valueOf(sensorOrientation));
                                      showInference(lastProcessingTimeMs + "ms");
                                    }
                                  });
                        }
                      }
                      } catch (IOException e) {
                        e.printStackTrace();
                      } finally {
                        if(assetInStream!=null) {
                          try {
                            assetInStream.close();
                          } catch (IOException e) {
                            e.printStackTrace();
                          }
                        }
                      }
                    readyForNextImage();
                    }
                });

  }

  @Override
  protected void onInferenceConfigurationChanged() {
    if (rgbFrameBitmap == null) {
      // Defer creation until we're getting camera frames.
      return;
    }
    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    runInBackground(() -> recreateClassifier(model, device, numThreads));
  }

  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU
        && (model == Model.QUANTIZED_MOBILENET || model == Model.QUANTIZED_EFFICIENTNET)) {
      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, R.string.tfe_ic_gpu_quant_error, Toast.LENGTH_LONG).show();
          });
      return;
    }
    try {
      LOGGER.d(
          "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create classifier.");
    }

    // Updates the input image size.
    imageSizeX = classifier.getImageSizeX();
    imageSizeY = classifier.getImageSizeY();
  }
}
