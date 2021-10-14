package com.ezafebrian.camerax_realtime_facerecognition.detector

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Point
import com.ezafebrian.camerax_realtime_facerecognition.util.*
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.MODEL_FILE_ONET
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.MODEL_FILE_PNET
import com.ezafebrian.camerax_realtime_facerecognition.util.Constant.MODEL_FILE_RNET
import org.tensorflow.lite.Interpreter
import java.util.*
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class MTCNN(assetManager: AssetManager) {

    private val options = Interpreter.Options().apply { setNumThreads(4) }

    private val pInterpreter: Interpreter =
        Interpreter(loadModelFile(assetManager, MODEL_FILE_PNET), options)

    private val rInterpreter: Interpreter =
        Interpreter(loadModelFile(assetManager, MODEL_FILE_RNET), options)

    private val oInterpreter: Interpreter =
        Interpreter(loadModelFile(assetManager, MODEL_FILE_ONET), options)

    private val factor = 0.709f
    private val pNetThreshold = 0.6f
    private val rNetThreshold = 0.7f
    private val oNetThreshold = 0.7f

    fun detectFaces(bitmap: Bitmap, minFaceSize: Int): Vector<Box> {
        var boxes: Vector<Box>
        try {
            // pNet
            boxes = pNet(bitmap, minFaceSize)
            squareLimit(boxes, bitmap.width, bitmap.height)

            //rNet
            boxes = rNet(bitmap, boxes)
            squareLimit(boxes, bitmap.width, bitmap.height)

            //oNet
            boxes = oNet(bitmap, boxes)
        } catch (e: IllegalStateException) {
            e.printStackTrace()
            boxes = Vector()
        }
        return boxes
    }

    private fun oNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {

        // ONet Input Init
        val num = boxes.size
        val oNetIn = Array(num) { Array(48) { Array(48) { FloatArray(3) } } }
        for (i in 0 until num) {
            var curCrop = cropAndResize(bitmap, boxes[i], 48)
            curCrop = transposeImage(curCrop)
            oNetIn[i] = curCrop
        }
        oNetForward(oNetIn, boxes)
        // ONetThreshold
        for (i in 0 until num) {
            if (boxes[i].score < oNetThreshold) {
                boxes[i].deleted = true
            }
        }
        boundingBoxRegression(boxes)
        nms(boxes, 0.7f, "Min")
        return updateBoxes(boxes)
    }

    private fun oNetForward(oNetIn: Array<Array<Array<FloatArray>>>, boxes: Vector<Box>) {
        val num = oNetIn.size
        val prob1 = Array(num) { FloatArray(2) }
        val conv62Conv62 = Array(num) { FloatArray(4) }
        val conv63Conv63 = Array(num) { FloatArray(10) }

        val outputs: MutableMap<Int, Any> = HashMap()
        outputs[oInterpreter.getOutputIndex("onet/prob1")] = prob1
        outputs[oInterpreter.getOutputIndex("onet/conv6-2/conv6-2")] = conv62Conv62
        outputs[oInterpreter.getOutputIndex("onet/conv6-3/conv6-3")] = conv63Conv63
        oInterpreter.runForMultipleInputsOutputs(arrayOf<Any>(oNetIn), outputs)

        for (i in 0 until num) {
            // prob
            boxes[i].score = prob1[i][1]
            // bias
            for (j in 0..3) {
                boxes[i].bbr[j] = conv62Conv62[i][j]
            }
            // landmark
            for (j in 0..4) {
                val x = (boxes[i].left() + conv63Conv63[i][j] * boxes[i].width()).roundToInt()
                val y =
                    (boxes[i].top() + conv63Conv63[i][j + 5] * boxes[i].height()).roundToInt()
                boxes[i].landmark[j] = Point(x, y)
            }
        }
    }

    private fun pNet(bitmap: Bitmap, minSize: Int): Vector<Box> {
        val whMin = min(bitmap.width, bitmap.height)
        var currentFaceSize = minSize.toFloat()
        val totalBoxes = Vector<Box>()
        while (currentFaceSize <= whMin) {
            val scale = 12.0f / currentFaceSize
            val bm = bitmapResize(bitmap, scale)
            val w = bm.width
            val h = bm.height
            val outW = (ceil(w * 0.5 - 5) + 0.5).toInt()
            val outH = (ceil(h * 0.5 - 5) + 0.5).toInt()
            var prob1 = Array(1) { Array(outW) { Array(outH) { FloatArray(2) } } }
            var conv42Biasadd = Array(1) { Array(outW) { Array(outH) { FloatArray(4) } } }
            pNetForward(bm, prob1, conv42Biasadd)
            prob1 = transposeBatch(prob1)
            conv42Biasadd = transposeBatch(conv42Biasadd)
            val curBoxes = Vector<Box>()
            generateBoxes(prob1, conv42Biasadd, scale, curBoxes)
            nms(curBoxes, 0.5f, "Union")
            for (i in curBoxes.indices) if (!curBoxes[i].deleted) totalBoxes.addElement(curBoxes[i])
            currentFaceSize /= factor
        }
        nms(totalBoxes, 0.7f, "Union")
        boundingBoxRegression(totalBoxes)
        return updateBoxes(totalBoxes)
    }

    private fun rNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        val num = boxes.size
        val rNetIn = Array(num) { Array(24) { Array(24) { FloatArray(3) } } }
        for (i in 0 until num) {
            var curCrop = cropAndResize(bitmap, boxes[i], 24)
            curCrop = transposeImage(curCrop)
            rNetIn[i] = curCrop
        }
        rNetForward(rNetIn, boxes)

        // RNetThreshold
        for (i in 0 until num) {
            if (boxes[i].score < rNetThreshold) {
                boxes[i].deleted = true
            }
        }
        nms(boxes, 0.7f, "Union")
        boundingBoxRegression(boxes)
        return updateBoxes(boxes)
    }

    private fun rNetForward(rNetIn: Array<Array<Array<FloatArray>>>, boxes: Vector<Box>) {
        val num = rNetIn.size
        val prob1 = Array(num) { FloatArray(2) }
        val conv52Conv52 = Array(num) { FloatArray(4) }

        val outputs: MutableMap<Int, Any> = HashMap()
        outputs[rInterpreter.getOutputIndex("rnet/prob1")] = prob1
        outputs[rInterpreter.getOutputIndex("rnet/conv5-2/conv5-2")] = conv52Conv52
        rInterpreter.runForMultipleInputsOutputs(arrayOf<Any>(rNetIn), outputs)

        for (i in 0 until num) {
            boxes[i].score = prob1[i][1]
            for (j in 0..3) {
                boxes[i].bbr[j] = conv52Conv52[i][j]
            }
        }
    }

    private fun squareLimit(boxes: Vector<Box>, w: Int, h: Int) {
         for (i in boxes.indices) {
            boxes[i].toSquareShape()
            boxes[i].limitSquare(w, h)
        }
    }

    private fun generateBoxes(
        prob1: Array<Array<Array<FloatArray>>>,
        conv4_2_BiasAdd: Array<Array<Array<FloatArray>>>,
        scale: Float,
        boxes: Vector<Box>
    ): Int {
        val h: Int = prob1[0].size
        val w: Int = prob1[0][0].size

        for (y in 0 until h) {
            for (x in 0 until w) {
                val score = prob1[0][y][x][1]
                if (score > pNetThreshold) {
                    val box = Box()
                    box.score = score
                    box.box[0] = (x * 2 / scale).roundToInt()
                    box.box[1] = (y * 2 / scale).roundToInt()
                    box.box[2] = ((x * 2 + 11) / scale).roundToInt()
                    box.box[3] = ((y * 2 + 11) / scale).roundToInt()
                    for (i in 0..3) {
                        box.bbr[i] = conv4_2_BiasAdd[0][y][x][i]
                    }
                    boxes.addElement(box)
                }
            }
        }
        return 0
    }

    private fun nms(boxes: Vector<Box>, threshold: Float, method: String) {
        for (i in 0 until boxes.size) {
            val box = boxes[i]
            if (!box.deleted) {
                for (j in i + 1 until boxes.size) {
                    val box2 = boxes[j]
                    if (!box2.deleted) {
                        val x1 = max(box.box[0], box2.box[0])
                        val y1 = max(box.box[1], box2.box[1])
                        val x2 = min(box.box[2], box2.box[2])
                        val y2 = min(box.box[3], box2.box[3])
                        if (x2 < x1 || y2 < y1) continue
                        val areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1)
                        var iou = 0f
                        if (method == "Union") iou =
                            1.0f * areaIoU / (box.area() + box2.area() - areaIoU) else if (method == "Min") iou =
                            1.0f * areaIoU / min(box.area(), box2.area())
                        if (iou >= threshold) {
                            if (box.score > box2.score) box2.deleted = true else box.deleted = true
                        }
                    }
                }
            }
        }
    }

    private fun boundingBoxRegression(boxes: Vector<Box>) {
        for (i in 0 until boxes.size) {
            boxes[i].calibrate()
        }
    }

    private fun updateBoxes(boxes: Vector<Box>): Vector<Box> {
        val b = Vector<Box>()
        for (i in 0 until boxes.size) {
            if (!boxes[i].deleted) {
                b.addElement(boxes[i])
            }
        }
        return b
    }

    private fun pNetForward(
        bitmap: Bitmap,
        prob1: Array<Array<Array<FloatArray>>>,
        conv4_2_BiasAdd: Array<Array<Array<FloatArray>>>
    ) {
        val img = normalizeImage(bitmap)
        var pNetIn: Array<Array<Array<FloatArray>>> = Array(1) { arrayOf() }
        pNetIn[0] = img
        pNetIn = transposeBatch(pNetIn)

        val outputs: MutableMap<Int, Any> = HashMap()
        outputs[pInterpreter.getOutputIndex("pnet/prob1")] = prob1
        outputs[pInterpreter.getOutputIndex("pnet/conv4-2/BiasAdd")] = conv4_2_BiasAdd

        pInterpreter.runForMultipleInputsOutputs(arrayOf<Any>(pNetIn), outputs)
    }
}