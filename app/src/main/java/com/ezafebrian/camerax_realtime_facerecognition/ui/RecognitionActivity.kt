package com.ezafebrian.camerax_realtime_facerecognition.ui

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.TextView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.ezafebrian.camerax_realtime_facerecognition.R
import com.ezafebrian.camerax_realtime_facerecognition.detector.Align
import com.ezafebrian.camerax_realtime_facerecognition.detector.MTCNN
import com.ezafebrian.camerax_realtime_facerecognition.recognition.MobileFaceNet
import com.ezafebrian.camerax_realtime_facerecognition.util.*
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.PHOTO_HEIGHT
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.PHOTO_WIDTH
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.THRESHOLD
import java.io.File
import java.lang.IllegalStateException
import java.lang.NullPointerException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class RecognitionActivity : AppCompatActivity() {
    companion object {
        const val TAG = "RecognitionActivity.kt"
    }

    private lateinit var mtcnn: MTCNN
    private lateinit var mobileFaceNet: MobileFaceNet
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var previewView: PreviewView
    private lateinit var textInfo: TextView

    private var registeredFace: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hideStatusBarAndNavigation()
        setContentView(R.layout.activity_recognition)

        previewView = findViewById(R.id.preview_view)
        textInfo = findViewById(R.id.text_info)

        mtcnn = MTCNN(assets)
        mobileFaceNet = MobileFaceNet(assets)

        cameraExecutor = Executors.newSingleThreadExecutor()

        prepareRegisteredFace()
        startCamera()
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(PHOTO_WIDTH, PHOTO_HEIGHT))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, { imageProxy ->
                        val image = imageProxy.image
                        image?.let { processImage(toBitmap(image)) }
                        imageProxy.close()
                    })
                }

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(bitmap: Bitmap) {
        if (registeredFace == null) return

        val rotatedBitmap = rotateBitmap(bitmap, -90f)

        try {
            val realtimeFace: Bitmap = cropImage(rotatedBitmap, mtcnn)
            try {
                val conf = mobileFaceNet.compare(registeredFace!!, realtimeFace)

                runOnUiThread {
                    if (conf > THRESHOLD) {
                        textInfo.setTextColor(Color.GREEN)
                        textInfo.text = "Same person: True, conf: $conf"
                    } else {
                        textInfo.setTextColor(Color.RED)
                        textInfo.text = "Same person: False, conf: $conf"
                    }
                }
            } catch (e: IllegalStateException) {
                err(e)
            }
        } catch (e: NullPointerException) {
            err(e)
        } catch (e: ArrayIndexOutOfBoundsException) {
            err(e)
        } catch (e: IllegalArgumentException) {
            err(e)
        }
    }

    private fun err(e: Throwable) {
        Log.e(TAG, e.stackTraceToString())
    }

    private fun hideStatusBarAndNavigation() {
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_IMMERSIVE
                or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN)
    }

    private fun prepareRegisteredFace() {
        if (registeredFace == null) registeredFace = cropRegisteredImage()
    }

    private fun cropRegisteredImage(): Bitmap {
        val file = File(getOutputDirectory(this), "photos.jpg")
        var bitmap = BitmapFactory.decodeFile(file.path)
        bitmap = rotateBitmap(bitmap, -90f)
        return cropImage(bitmap, mtcnn)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}