package com.ezafebrian.camerax_realtime_facerecognition.detector

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Point
import kotlin.math.abs
import kotlin.math.atan

class Align {
    fun faceAlign(bitmap:Bitmap, landmarks:Array<Point?>): Bitmap? {
        val diffEyeX = (landmarks[1]!!.x - landmarks[0]!!.x).toFloat()
        val diffEyeY = (landmarks[1]!!.y - landmarks[0]!!.y).toFloat()

        val fAngle: Float = if (abs(diffEyeY) < 1e-7) {
            0f
        } else {
            (atan((diffEyeY / diffEyeX).toDouble()) * 180.0f / Math.PI).toFloat()
        }
        val matrix = Matrix()
        matrix.setRotate(-fAngle)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}