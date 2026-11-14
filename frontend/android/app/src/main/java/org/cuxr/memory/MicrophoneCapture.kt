package org.cuxr.memory

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.AtomicFile
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.locks.ReentrantLock

/** A bounded local recording for testing microphone access on the glasses. */
class MicrophoneCapture(directory: File) {
    private val latest = AtomicFile(File(directory, "microphone.wav"))

    @Volatile
    private var stopping = false

    fun stop() {
        stopping = true
    }

    fun record(onStarted: () -> Unit): Int {
        check(recordingLock.tryLock()) { "Previous recording still closing" }
        try {
            return recordLocked(onStarted)
        } finally {
            recordingLock.unlock()
        }
    }

    @SuppressLint("MissingPermission") // The activity checks RECORD_AUDIO before starting.
    private fun recordLocked(onStarted: () -> Unit): Int {
        val bufferSize = AudioRecord.getMinBufferSize(RATE, CHANNEL, ENCODING)
        check(bufferSize > 0) { "16 kHz mono PCM unavailable" }
        val samples = ShortArray(RATE / 10)
        val pcm = ByteBuffer.allocate(samples.size * BYTES_PER_SAMPLE).order(ByteOrder.LITTLE_ENDIAN)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC, RATE, CHANNEL, ENCODING, maxOf(bufferSize, pcm.capacity()),
        )
        try {
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "Microphone initialization failed" }
            if (stopping) return 0
            recorder.startRecording()
            check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "Microphone did not start" }
            val output = latest.startWrite()
            try {
                output.write(ByteArray(44))
                var bytes = 0
                onStarted()
                while (!stopping && bytes < MAX_BYTES) {
                    val count = recorder.read(samples, 0, minOf(samples.size, (MAX_BYTES - bytes) / BYTES_PER_SAMPLE))
                    if (count <= 0) throw IOException("Microphone read failed ($count)")
                    pcm.clear()
                    pcm.asShortBuffer().put(samples, 0, count)
                    val length = count * BYTES_PER_SAMPLE
                    output.write(pcm.array(), 0, length)
                    bytes += length
                }
                if (bytes == 0) {
                    latest.failWrite(output)
                    return 0
                }
                output.channel.position(0)
                output.write(wavHeader(bytes))
                latest.finishWrite(output)
                return bytes / BYTES_PER_SAMPLE
            } catch (error: Exception) {
                latest.failWrite(output)
                throw error
            }
        } finally {
            recorder.release()
        }
    }

    private fun wavHeader(bytes: Int): ByteArray =
        ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            .put("RIFF".toByteArray()).putInt(bytes + 36)
            .put("WAVEfmt ".toByteArray()).putInt(16)
            .putShort(1).putShort(1).putInt(RATE)
            .putInt(RATE * BYTES_PER_SAMPLE).putShort(BYTES_PER_SAMPLE.toShort()).putShort(16)
            .put("data".toByteArray()).putInt(bytes)
            .array()

    companion object {
        private val recordingLock = ReentrantLock()
        private const val RATE = 16_000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        private const val BYTES_PER_SAMPLE = 2
        private const val MAX_BYTES = RATE * BYTES_PER_SAMPLE * 60
    }
}
