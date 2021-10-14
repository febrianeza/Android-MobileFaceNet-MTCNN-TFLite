package com.ezafebrian.camerax_realtime_facerecognition.recognition

import android.content.res.AssetManager
import android.graphics.Bitmap
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.INPUT_IMAGE_SIZE
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.MOBILE_FACE_NET_MODEL
import com.ezafebrian.camerax_realtime_facerecognition.util.l2Normalize
import com.ezafebrian.camerax_realtime_facerecognition.util.loadModelFile
import com.ezafebrian.camerax_realtime_facerecognition.util.normalizeImage
import org.tensorflow.lite.Interpreter
import kotlin.math.pow

class MobileFaceNet(assetManager: AssetManager) {
    private val options = Interpreter.Options().apply { setNumThreads(4) }
    private val interpreter: Interpreter =
        Interpreter(loadModelFile(assetManager, MOBILE_FACE_NET_MODEL), options)

    fun compare(bitmap1: Bitmap, bitmap2: Bitmap): Float {
        val bitmapScale1 =
            Bitmap.createScaledBitmap(bitmap1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true)
        val bitmapScale2 =
            Bitmap.createScaledBitmap(bitmap2, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true)

        val datasets = getTwoImageDatasets(bitmapScale1, bitmapScale2)
        val embeddings = Array(2) { FloatArray(192) }

        interpreter.run(datasets, embeddings)
        l2Normalize(embeddings, 1e-10)
        return evaluate(embeddings)
    }

    private fun evaluate(embeddings: Array<FloatArray>): Float {
        val embeddings1 = embeddings[0]
        val embeddings2 = embeddings[1]
        var dist = 0f
        for (i in 0..191) {
            dist += (embeddings1[i] - embeddings2[i]).toDouble().pow(2.0).toFloat()
        }
        var same = 0f
        for (i in 0..399) {
            val threshold = 0.01f * (i + 1)
            if (dist < threshold) {
                same += (1.0 / 400).toFloat()
            }
        }
        return same
    }

    private fun getTwoImageDatasets(
        bitmap1: Bitmap,
        bitmap2: Bitmap
    ): Array<Array<Array<FloatArray>>> {
        val bitmaps = arrayOf(bitmap1, bitmap2)

        val ddims = intArrayOf(
            bitmaps.size,
            INPUT_IMAGE_SIZE,
            INPUT_IMAGE_SIZE,
            3
        )
        val datasets =
            Array(ddims[0]) { Array(ddims[1]) { Array(ddims[2]) { FloatArray(ddims[3]) } } }

        for (i in 0 until ddims[0]) {
            val bitmap = bitmaps[i]
            datasets[i] = normalizeImage(bitmap)
        }
        return datasets
    }
}