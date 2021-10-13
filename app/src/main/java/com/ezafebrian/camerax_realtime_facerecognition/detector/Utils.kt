package com.ezafebrian.camerax_realtime_facerecognition.detector

import android.graphics.*
import android.util.Log
import java.io.IOException
import java.util.*
import kotlin.math.max
import kotlin.math.min

object Utils {
    fun copyBitmap(bitmap: Bitmap): Bitmap = bitmap.copy(bitmap.config, true)

    fun drawRect(bitmap: Bitmap, rect: Rect, thick: Int) {
        try {
            val canvas = Canvas(bitmap)
            val r = 255
            val g = 0
            val b = 0
            val paint = Paint().apply {
                color = Color.rgb(r, g, b)
                strokeWidth = thick.toFloat()
                style = Paint.Style.STROKE
            }
            canvas.drawRect(rect, paint)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun drawPoints(bitmap: Bitmap, landmark: Array<Point?>, thick: Int) {
        landmark.forEach {
            it?.let {
                val x = it.x
                val y = it.y
                drawRect(bitmap, Rect(x - 1, y - 1, x + 1, y + 1), thick)
            }
        }
    }

    fun drawBox(bitmap: Bitmap, box: Box, thick: Int) {
        drawRect(bitmap, box.transform2Rect(), thick)
        drawPoints(bitmap, box.landmark, thick)
    }

    fun flipDiag(data: FloatArray, h: Int, w: Int, stride: Int) {
        val tmp = FloatArray(w * h * stride)
        for (i in 0 until w * h * stride) tmp[i] = data[i]
        for (y in 0 until h) for (x in 0 until w) {
            for (z in 0 until stride) data[(x * h + y) * stride + z] = tmp[(y * w + x) * stride + z]
        }
    }

    fun expand(src: FloatArray, dst: Array<FloatArray>) {
        var idx = 0
        for (y in dst.indices)
            for (x in dst[0].indices)
                dst[y][x] = src[idx++]
    }

    fun expand(src: FloatArray, dst: Array<Array<FloatArray>>) {
        var idx = 0
        for (y in dst.indices)
            for (x in dst[0].indices)
                for (c in dst[0][0].indices)
                    dst[y][x][c] = src[idx++]
    }

    fun expandProb(src: FloatArray, dst: Array<FloatArray>) {
        var idx = 0
        for (y in dst.indices)
            for (x in dst[0].indices)
                dst[y][x] = src[idx++ * 2 + 1]
    }

    fun boxes2rects(boxes: Vector<Box>): Array<Rect?> {
        var cnt = 0
        for (i in boxes.indices) if (!boxes[i].deleted) cnt++
        val r = arrayOfNulls<Rect>(cnt)
        var idx = 0
        for (i in boxes.indices) if (!boxes[i].deleted) r[idx++] = boxes[i].transform2Rect()
        return r
    }

    fun updateBoxes(boxes: Vector<Box>): Vector<Box> {
        val b = Vector<Box>()
        for (i in boxes.indices) if (!boxes[i].deleted) b.addElement(boxes[i])
        return b
    }

    fun crop(bitmap: Bitmap, rect: Rect): Bitmap {
        return Bitmap.createBitmap(
            bitmap,
            rect.left,
            rect.top,
            rect.right - rect.left,
            rect.bottom - rect.top
        )
    }

    fun rectExtend(bitmap: Bitmap, rect: Rect, pixels: Int) {
        rect.left = max(0, rect.left - pixels)
        rect.right = min(bitmap.width - 1, rect.right + pixels)
        rect.top = max(0, rect.top - pixels)
        rect.bottom = min(bitmap.height - 1, rect.bottom + pixels)
    }

    fun showPixel(v:Int) {
        Log.i(
            "MainActivity",
            "[*]Pixel:R" + (v shr 16 and 0xff) + "G:" + (v shr 8 and 0xff) + " B:" + (v and 0xff)
        )
    }

    fun resize(_bitmap: Bitmap, new_width: Int): Bitmap {
        val scale = new_width.toFloat() / _bitmap.width
        val matrix = Matrix()
        matrix.postScale(scale, scale)
        return Bitmap.createBitmap(_bitmap, 0, 0, _bitmap.width, _bitmap.height, matrix, true)
    }
}