package com.ezafebrian.camerax_realtime_facerecognition.detector

import android.graphics.Point
import android.graphics.Rect
import kotlin.math.max

class Box {
    var box: IntArray = IntArray(4)
    var score: Float = 0f
    var bbr: FloatArray = FloatArray(4)
    var deleted: Boolean = false
    var landmark: Array<Point?> = arrayOfNulls(5)

    fun left(): Int = box[0]
    fun right(): Int = box[1]
    fun top(): Int = box[2]
    fun bottom(): Int = box[3]

    fun width(): Int = box[2] - box[0] + 1
    fun height(): Int = box[3] - box[1] + 1

    fun transform2Rect(): Rect {
        val rect = Rect()
        rect.left = box[0]
        rect.top = box[1]
        rect.right = box[2]
        rect.bottom = box[3]
        return rect
    }

    fun area(): Int = width() * height()

    fun calibrate() {
        val w = box[2] - box[0] + 1
        val h = box[3] - box[1] + 1
        box[0] = (box[0] + w * bbr[0]).toInt()
        box[1] = (box[1] + h * bbr[1]).toInt()
        box[2] = (box[2] + w * bbr[2]).toInt()
        box[3] = (box[3] + h * bbr[3]).toInt()
        for (i in 0..3) bbr[i] = 0.0f
    }

    fun toSquareShape() {
        val w = width()
        val h = height()
        if (w > h) {
            box[1] -= (w - h) / 2
            box[3] += (w - h + 1) / 2
        } else {
            box[0] -= (h - w) / 2
            box[2] += (h - w + 1) / 2
        }
    }

    fun limitSquare(w: Int, h: Int) {
        if (box[0] < 0 || box[1] < 0) {
            val len = max(-box[0], -box[1])
            box[0] += len
            box[1] += len
        }
        if (box[2] >= w || box[3] >= h) {
            val len = max(box[2] - w + 1, box[3] - h + 1)
            box[2] -= len
            box[3] -= len
        }
    }

    fun transbound(w: Int, h: Int): Boolean {
        if (box[0] < 0 || box[1] < 0) {
            return true
        } else if (box[2] >= w || box[3] >= h) {
            return true
        }
        return false
    }
}