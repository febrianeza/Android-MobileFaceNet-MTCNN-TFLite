package com.ezafebrian.camerax_realtime_facerecognition.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.ProgressBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatImageView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.ezafebrian.camerax_realtime_facerecognition.R
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.PHOTO_HEIGHT
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.PHOTO_WIDTH
import com.ezafebrian.camerax_realtime_facerecognition.util.getOutputDirectory
import java.io.File

class RegistrationActivity : AppCompatActivity() {
    companion object {
        const val TAG = "RegistrationActivity.kt"
    }

    private lateinit var progressBar: ProgressBar
    private lateinit var buttonCapture: AppCompatImageView
    private lateinit var cameraPreview: PreviewView

    private var imageCapture: ImageCapture? = null

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            startCamera()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hideStatusBarAndNavigation()
        setContentView(R.layout.activity_registration)

        progressBar = findViewById(R.id.progress_bar)
        buttonCapture = findViewById(R.id.capture_camera)
        cameraPreview = findViewById(R.id.preview_view)

        buttonCapture.setOnClickListener {
            showLoading()
            takePhoto()
        }

        if (!allPermissionsGranted()) {
            permissionLauncher.launch(
                arrayOf(
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                )
            )
        } else {
            startCamera()
        }
    }

    private fun hideStatusBarAndNavigation() {
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_IMMERSIVE
                or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN)
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            imageCapture = ImageCapture.Builder()
                .setTargetResolution(Size(PHOTO_WIDTH, PHOTO_HEIGHT))
                .build()

            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(cameraPreview.surfaceProvider)
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        val photoFile = File(
            getOutputDirectory(this),
            "photos.jpg"
        )

        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    hideLoading()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    hideLoading()
                    startActivity(
                        Intent(
                            this@RegistrationActivity,
                            RecognitionActivity::class.java
                        )
                    )
                }
            })
    }

    private fun allPermissionsGranted(): Boolean {
        return ActivityCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
            this, Manifest.permission.WRITE_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun showLoading() {
        progressBar.visibility = View.VISIBLE
        buttonCapture.isEnabled = false
    }

    private fun hideLoading() {
        progressBar.visibility = View.INVISIBLE
        buttonCapture.isEnabled = true
    }

    override fun onResume() {
        super.onResume()
        startCamera()
    }
}